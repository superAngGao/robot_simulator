"""Run the actual scalar predicates on captured edge cases, without a GPU.

Warp's Python bodies are evaluated after removing only decorators/annotations.
CUDA's original C++ helper is compiled as host C++, when a compiler is present.
These checks do not replace native compilation and supervised Go2 readback.
"""
import ast
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from optics.execution import _intersect_triangle_scalar
from optics import _cuda_direct_light_source


ROOT = Path(__file__).resolve().parents[3]
CASES = json.loads((ROOT / "tests/data/optics/go2_intersection_edge_repros_v1.json").read_text())["cases"]


def _warp_bodies():
    names = {"_intersect_device_triangle_for_ray", "_is_better_hit",
             "_intersect_aabb_axis", "_intersect_aabb_for_ray_with_interval",
             "_device_scene_bvh_first_hit_kernel"}
    tree = ast.parse((ROOT / "optics/warp_native_render.py").read_text())
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names and node.decorator_list:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            functions.append(node)
    assert {f.name for f in functions} == names
    wp = SimpleNamespace(float32=np.float32, float64=np.float64, int32=np.int32,
                         int64=np.int64, abs=abs, min=min, inf=np.inf, nan=np.nan,
                         tid=lambda: 0, zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype),
                         atomic_add=lambda a, i, v: a.__setitem__(i, a[i] + v),
                         atomic_max=lambda a, i, v: a.__setitem__(i, max(a[i], v)))
    namespace = {"wp": wp, "_BUILD_EPS": 1e-8, "_DIR_EPS": 1e-8,
                 "_T_EPS": 1e-5, "_MAX_BVH_STACK": 64}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_warp_bodies", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def warp_predicate():
    body = _warp_bodies()["_intersect_device_triangle_for_ray"]

    def evaluate(origin, direction, v0, e1, e2, limit):
        return body(*origin, *direction, 0, v0[None], e1[None], e2[None],
                    np.zeros((1, 3), np.float32), np.float32(limit))[:2]
    return evaluate


@pytest.fixture(scope="module")
def cuda_host(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable; native GPU validation remains required")
    source = _cuda_direct_light_source.CUDA_SOURCE
    start = source.index("__device__ __forceinline__ bool better_hit(")
    end = source.index("\n}", source.index("__global__ void first_hit_rays_kernel(")) + 2
    body = source[start:end].replace("__device__ __forceinline__", "")
    body = body.replace("__global__ void", 'extern "C" void')
    directory = tmp_path_factory.mktemp("actual_cuda_predicate")
    cpp, library = directory / "predicate.cpp", directory / "predicate.so"
    cpp.write_text('''#include <cmath>
#include <cstdint>
#include <limits>
constexpr float kBuildEps=1e-8f, kDirEps=1e-8f, kTEps=1e-5f;
constexpr int kMaxBvhStack=64;
struct Index { int x; };
constexpr Index blockIdx{0}, blockDim{1}, threadIdx{0};
void atomicAdd(int32_t* p, int v) { *p += v; }
void atomicMax(int32_t* p, int v) { if (v > *p) *p = v; }
''' + body + '''
extern "C" int evaluate(const float* o, const float* d, const float* b,
    const float* e1, const float* e2, float limit, float* t) {
  const float n[3]={0,0,0}; float nx,ny,nz;
  return intersect_triangle(o[0],o[1],o[2],d[0],d[1],d[2],0,b,e1,e2,n,
                            limit,t,&nx,&ny,&nz);
}
''')
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(cpp), "-o", str(library)],
                   check=True, capture_output=True, timeout=60)
    dll = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    dll.evaluate.argtypes = [pointer] * 5 + [ctypes.c_float, pointer]
    dll.evaluate.restype = ctypes.c_int

    def evaluate(origin, direction, v0, e1, e2, limit):
        t = ctypes.c_float()
        hit = dll.evaluate(*(a.ctypes.data_as(pointer) for a in (origin, direction, v0, e1, e2)),
                           limit, ctypes.byref(t))
        return bool(hit), t.value
    def first_hit(*args):
        native = []
        for value in (*args, 1):
            if isinstance(value, np.ndarray):
                native.append(value.ctypes.data_as(ctypes.POINTER(np.ctypeslib.as_ctypes_type(value.dtype))))
            elif isinstance(value, np.floating):
                native.append(ctypes.c_float(value))
            elif isinstance(value, np.int64):
                native.append(ctypes.c_int64(value))
            else:
                native.append(ctypes.c_int(value))
        dll.first_hit_rays_kernel.argtypes = [type(v) for v in native]
        dll.first_hit_rays_kernel.restype = None
        dll.first_hit_rays_kernel(*native)
    return SimpleNamespace(predicate=evaluate, first_hit=first_hit)


@pytest.fixture(scope="module")
def cuda_predicate(cuda_host):
    return cuda_host.predicate


@pytest.mark.parametrize("backend", ["warp_predicate", "cuda_predicate"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['lane']}-{c['pixel']}-{c['primitive']}")
def test_captured_edges_match_original_cpu_oracle(request, backend, case):
    evaluate = request.getfixturevalue(backend)
    origin, direction, v0, e1, e2 = (np.asarray(case[name], np.float32)
                                    for name in ("origin", "direction", "v0", "e1", "e2"))
    b = v0.astype(np.float64)
    truth = _intersect_triangle_scalar(origin.astype(np.float64), direction.astype(np.float64),
                                      10.0, b, b + e1.astype(np.float64), b + e2.astype(np.float64))
    hit, distance = evaluate(origin, direction, v0, e1, e2, 10.0)
    assert bool(hit) == (truth is not None)
    if truth is not None:
        assert distance == np.float32(truth.distance)
        # Range clipping remains strict; increased precision is not an edge epsilon.
        limit = np.nextafter(np.float32(truth.distance), np.float32(0))
        assert not evaluate(origin, direction, v0, e1, e2, limit)[0]


@pytest.mark.parametrize("separate_leaves", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("backend", ["warp", "cuda"])
def test_primary_traversal_keeps_candidates_within_existing_tie_window(request, backend, separate_leaves, reverse):
    body = (_warp_bodies()["_device_scene_bvh_first_hit_kernel"] if backend == "warp"
            else request.getfixturevalue("cuda_host").first_hit)
    # The slightly farther face wins by the existing source-key rule. It must
    # survive both leaf intersection clipping and node/child AABB pruning.
    v0 = np.array([[1, -1, -1], [1.000005, -1, -1]], np.float32)
    e1 = np.array([[0, 2, 0]] * 2, np.float32)
    e2 = np.array([[0, 0, 2]] * 2, np.float32)
    lo, hi = v0.copy(), v0 + e1 + e2
    ids = np.array([1, 0] if reverse else [0, 1], np.int32)
    if separate_leaves:
        bounds_min = np.vstack([lo.min(axis=0), lo])
        bounds_max = np.vstack([hi.max(axis=0), hi])
        left, right, start, count = [1, -1, -1], [2, -1, -1], [0, 0, 1], [0, 1, 1]
        if reverse:
            bounds_min[[1, 2]] = bounds_min[[2, 1]]
            bounds_max[[1, 2]] = bounds_max[[2, 1]]
    else:
        bounds_min, bounds_max = lo.min(axis=0)[None], hi.max(axis=0)[None]
        left, right, start, count = [-1], [-1], [0], [2]
    hit, distance, instance, material = (np.zeros(1, dtype=t) for t in (np.int32, np.float32, np.int32, np.int32))
    zero_i = np.empty(0, np.int32)
    zero_key = np.empty(0, np.int64)
    zero_vec = np.empty((0, 3), np.float32)
    body(np.zeros((1, 3), np.float32), np.array([[1, 0, 0]], np.float32), np.float32(10),
         zero_vec, zero_vec, zero_i, zero_key, zero_key, zero_i, 0,
         v0, e1, e2, np.array([[-1, 0, 0]] * 2, np.float32),
         np.array([20, 10], np.int32), np.array([20, 10], np.int64), np.ones(2, np.int64),
         np.array([2, 1], np.int32), bounds_min, bounds_max,
         *(np.array(v, np.int32) for v in (left, right, start, count)), ids, len(count), np.int64(1),
         hit, distance, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), instance, material,
         np.zeros(1, np.int32), np.zeros(1, np.int32))
    assert hit[0] == 1 and instance[0] == 10 and material[0] == 1
    assert distance[0] == v0[1, 0]
