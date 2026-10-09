"""Run the actual scalar predicates on captured edge cases, without a GPU.

Warp's Python bodies are evaluated after removing only decorators/annotations.
CUDA's original C++ helper is compiled as host C++, when a compiler is present.
These checks do not replace native compilation and supervised Go2 readback.
"""
import ast
import ctypes
import json
from pathlib import Path
import re
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
                         cast=lambda value, dtype: np.asarray(value).view(dtype)[()],
                         tid=lambda: 0, zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype),
                         atomic_add=lambda a, i, v: a.__setitem__(i, a[i] + v),
                         atomic_max=lambda a, i, v: a.__setitem__(i, max(a[i], v)))
    constants = {"_BUILD_EPS", "_DIR_EPS", "_T_EPS", "_MAX_BVH_STACK"}
    namespace = {"wp": wp}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in constants:
                namespace[name] = ast.literal_eval(node.value)
    assert constants <= namespace.keys()
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
struct Index { int x; };
constexpr Index blockIdx{0}, blockDim{1}, threadIdx{0};
void atomicAdd(int32_t* p, int v) { *p += v; }
void atomicMax(int32_t* p, int v) { if (v > *p) *p = v; }
''' + "\n".join(re.findall(r"^constexpr (?:float|int) (?:kBuildEps|kDirEps|kTEps|kMaxBvhStack) = .*;$",
                           source, re.MULTILINE)) + "\n" + body + '''
extern "C" int evaluate(const float* o, const float* d, const float* b,
    const float* e1, const float* e2, float limit, float* t) {
  const float n[3]={0,0,0}; float nx,ny,nz;
  return intersect_triangle(o[0],o[1],o[2],d[0],d[1],d[2],0,b,e1,e2,n,
                            limit,t,&nx,&ny,&nz);
}
extern "C" int evaluate_box(const float* o, const float* d, const float* lo,
    const float* hi, float limit, float* entry) {
  return intersect_aabb(o[0],o[1],o[2],d[0],d[1],d[2],lo,hi,0,limit,entry);
}
''')
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(cpp), "-o", str(library)],
                   check=True, capture_output=True, timeout=60)
    dll = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    dll.evaluate.argtypes = [pointer] * 5 + [ctypes.c_float, pointer]
    dll.evaluate.restype = ctypes.c_int
    dll.evaluate_box.argtypes = [pointer] * 4 + [ctypes.c_float, pointer]
    dll.evaluate_box.restype = ctypes.c_int

    def evaluate(origin, direction, v0, e1, e2, limit):
        t = ctypes.c_float()
        hit = dll.evaluate(*(a.ctypes.data_as(pointer) for a in (origin, direction, v0, e1, e2)),
                           limit, ctypes.byref(t))
        return bool(hit), t.value
    def box(origin, direction, lo, hi, limit):
        entry = ctypes.c_float()
        hit = dll.evaluate_box(*(a.ctypes.data_as(pointer) for a in (origin, direction, lo, hi)),
                               limit, ctypes.byref(entry))
        return bool(hit), entry.value
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
    return SimpleNamespace(predicate=evaluate, first_hit=first_hit, box=box)


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
@pytest.mark.parametrize("boundary", ["interior", "rounded_tie", "outside_tie", "physical_limit",
                                      "oblique", "oblique_outside", "oblique_limit", "half_ulp"])
def test_primary_traversal_keeps_candidates_within_existing_tie_window(request, backend, separate_leaves, reverse, boundary):
    body = (_warp_bodies()["_device_scene_bvh_first_hit_kernel"] if backend == "warp"
            else request.getfixturevalue("cuda_host").first_hit)
    # The slightly farther face wins by the existing source-key rule. It must
    # survive both leaf intersection clipping and node/child AABB pruning.
    near = np.float32(1 if boundary == "interior" else .1)
    far = np.float32(1.000005) if boundary == "interior" else np.float32(near + np.float32(1e-5))
    if boundary == "outside_tie":
        far = np.nextafter(far, np.float32(np.inf))
    origin = np.array([[0 if boundary == "interior" else -1e-9, 0, 0]], np.float32)
    direction = np.array([[1, 0, 0]], np.float32)
    max_distance = far if boundary == "physical_limit" else np.float32(10)
    expected_id = 20 if boundary in {"outside_tie", "physical_limit"} else 10
    v0 = np.array([[near, -1, -1], [far, -1, -1]], np.float32)
    e1 = np.array([[0, 2, 0]] * 2, np.float32)
    e2 = np.array([[0, 0, 2]] * 2, np.float32)
    expected_distance = far if expected_id == 10 else near
    if boundary == "half_ulp":
        origin[:] = [-3.725290298461914e-9, 0, 0]
        far = np.float32(0.1076514795422554)
        near = np.float32(far - np.float32(1e-5))
        v0[:, 0] = [near, far]
        e1[:] = [0, 3.5386428833007812, 0]
        e2[:] = [0, 0, 7.4439778327941895]
        expected_distance = far
    if boundary.startswith("oblique"):
        origin[:] = 0
        direction[:] = [0.8446270823478699, 0.5353550910949707, 0]
        near, far = np.float32(0.1518450379371643), np.float32(0.15185348689556122)
        if boundary == "oblique_outside":
            far = np.nextafter(far, np.float32(np.inf))
        v0[:] = [[near, -2, -2], [far, -2, -2]]
        e1[:] = [0, 8, 0]
        e2[:] = [0, 0, 8]
        expected_id = 10 if boundary == "oblique" else 20
        expected_distance = np.float32(np.float64(far if expected_id == 10 else near) / direction[0, 0])
        if boundary == "oblique_limit":
            # This rounded limit is just below the true far-plane distance.
            max_distance = np.float32(np.float64(far) / direction[0, 0])
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
    body(origin, direction, max_distance,
         zero_vec, zero_vec, zero_i, zero_key, zero_key, zero_i, 0,
         v0, e1, e2, np.array([[-1, 0, 0]] * 2, np.float32),
         np.array([20, 10], np.int32), np.array([20, 10], np.int64), np.ones(2, np.int64),
         np.array([2, 1], np.int32), bounds_min, bounds_max,
         *(np.array(v, np.int32) for v in (left, right, start, count)), ids, len(count), np.int64(1),
         hit, distance, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), instance, material,
         np.zeros(1, np.int32), np.zeros(1, np.int32))
    assert hit[0] == 1 and instance[0] == expected_id
    assert material[0] == (1 if expected_id == 10 else 2)
    assert distance[0] == expected_distance


@pytest.mark.parametrize("backend", ["warp_predicate", "cuda_predicate"])
@pytest.mark.parametrize("offset", [np.float32(-1e-9), np.float32(0), np.float32(1e-9)])
def test_physical_and_shadow_cutoff_stays_strict_before_distance_rounding(request, backend, offset):
    # The shared helper is used by primary and shadow rays. All three distances
    # round to the same ABI float, but the just-outside distance must reject.
    evaluate = request.getfixturevalue(backend)
    limit = np.float32(.1)
    origin = np.array([offset, 0, 0], np.float32)
    d = np.array([1, 0, 0], np.float32)
    b = np.array([limit, -1, -1], np.float32)
    e1, e2 = np.array([0, 2, 0], np.float32), np.array([0, 0, 2], np.float32)
    assert bool(evaluate(origin, d, b, e1, e2, limit)[0]) == (offset >= 0)
    # A genuine negative geometric distance stays a miss, even very near zero.
    b[0] = 0
    origin[0] = np.nextafter(np.float32(0), np.float32(1))
    assert not evaluate(origin, d, b, e1, e2, limit)[0]


@pytest.mark.parametrize("backend", ["warp", "cuda"])
def test_stack_entry_quantization_preserves_lower_bound(request, backend):
    if backend == "warp":
        body = _warp_bodies()["_intersect_aabb_for_ray_with_interval"]
        box = lambda o, d, lo, hi, limit: body(*o, *d, *lo, *hi, np.float32(limit))[:2]
        predicate = request.getfixturevalue("warp_predicate")
    else:
        host = request.getfixturevalue("cuda_host")
        box, predicate = host.box, host.predicate
    rng = np.random.default_rng(710)
    d = np.array([1, 0, 0], np.float32)
    for _ in range(5000):
        x = np.float32(rng.uniform(.08, .2))
        # Halfway distances exercise round-to-even on both parities, with
        # varying edges so the triangle calculation follows its actual body.
        o = np.array([-np.spacing(x) / np.float32(2), 0, 0], np.float32)
        b = np.array([x, -1, -1], np.float32)
        e1 = np.array([0, rng.uniform(3, 9), 0], np.float32)
        e2 = np.array([0, 0, rng.uniform(3, 9)], np.float32)
        hit, distance = predicate(o, d, b, e1, e2, 10)
        box_hit, entry = box(o, d, b, b + e1 + e2, 10)
        assert hit and box_hit
        assert entry <= np.float64(x) - np.float64(o[0])
        assert entry <= distance
