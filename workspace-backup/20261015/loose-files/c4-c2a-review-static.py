import ast
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
base = 'a1f6c8c87752559c7b25994d450a36c56c952b32'
def old(path):
    return subprocess.check_output(['git', 'show', f'{base}:{path}'], cwd=root, text=True, encoding='utf-8')
def parse(source):
    return ast.parse(source)
def meaningful(source):
    return [ast.dump(n, include_attributes=False) for n in parse(source).body
            if not isinstance(n, (ast.Import, ast.ImportFrom))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
for legacy, production in [('native','native_input'),('receipts','provenance'),('timing','timing'),('cases','cases'),('workload_contract','workload_contract')]:
    assert meaningful(old(f'benchmarks/optical/{legacy}.py')) == meaningful((root/f'optical_worker/{production}.py').read_text(encoding='utf-8')), legacy
    print('exact all non-import AST statements:',legacy)

def defs(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for path in ['optics/device_bvh.py','optics/device_scene.py','benchmarks/optical/eligibility.py','optics/cuda_arena.py']:
    before=defs(old(path)); after=defs((root/path).read_text(encoding='utf-8'))
    removed=set(before)-set(after)
    changed={k for k in before.keys()&after.keys() if before[k]!=after[k]}
    print(path,'removed=',sorted(removed),'changed=',sorted(changed))
    assert changed == ({'_canonical_c3_binding','_validate_c3_adapter_authority'} if path=='optics/cuda_arena.py' else set())
    assert set(after)<=set(before)

for path in ['optical_worker/native_contract.py','optical_worker/worker.py','optical_worker/manifest.py','optical_worker/protocol.py','optical_worker/native_setup.py','benchmarks/results/q54_p12_4/gate.json','benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json']:
    data=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=root)
    now=subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=root)
    assert data==now,path
    print('git blob bytes unchanged:',path)

init=parse((root/'optics/__init__.py').read_text(encoding='utf-8'))
assert [type(n).__name__ for n in init.body]==['Expr','Assign','FunctionDef','FunctionDef']
getattr_node=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='__getattr__')
seen={}
for n in getattr_node.body[:-1]:
    assert isinstance(n,ast.If) and not n.orelse
    assert ast.dump(n.test.left)=='Name(id=\'name\', ctx=Load())'
    name=ast.literal_eval(n.test.comparators[0])
    assert name not in seen
    assert len(n.body)==3 and isinstance(n.body[0],ast.ImportFrom)
    assert n.body[0].level==1 and len(n.body[0].names)==1
    seen[name]=(n.body[0].module,n.body[0].names[0].name)
before=parse(old('optics/__init__.py'))
expected={a.asname or a.name:(n.module,a.name) for n in before.body if isinstance(n,ast.ImportFrom) for a in n.names}
changes={key:(expected[key],seen[key]) for key in expected if seen.get(key)!=expected[key]}
assert set(seen)==set(expected)
assert changes == {name:(('device_bvh',name),('host_bvh',name)) for name in ['DeviceBvhBuildStats','build_host_median_bvh']},changes
print('static lazy branches preserve exact original import map:',len(seen))

before_lane={n.targets[0].id:ast.dump(n,include_attributes=False) for n in parse(old('benchmarks/optical/eligibility.py')).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
after_lane={n.targets[0].id:ast.dump(n,include_attributes=False) for n in parse((root/'optical_worker/lane_contract.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
assert len(after_lane)==7
assert all(before_lane[name]==value for name,value in after_lane.items())
print('all seven lane identity assignments unchanged')

sys.path.insert(0,str(root))
import gc
import weakref
from optical_worker import provenance as production
from benchmarks.optical import receipts as legacy
assert production._CUDA_ARENA_C3_BINDING_AUTHORITIES is legacy._CUDA_ARENA_C3_BINDING_AUTHORITIES
binding=production.C3OneShotNativeExecutionBindingEvidence.__new__(production.C3OneShotNativeExecutionBindingEvidence)
identity=id(binding)
target,prepared,readiness=object(),object(),object()
production._register_cuda_arena_c3_binding_authority(binding,target=target,prepared_input=prepared,native_readiness=readiness)
legacy._assert_cuda_arena_c3_binding_authority(binding,target=target,prepared_input=prepared,native_readiness=readiness)
for key in ('target','prepared_input','native_readiness'):
    args=dict(target=target,prepared_input=prepared,native_readiness=readiness)
    args[key]=object()
    try:
        legacy._assert_cuda_arena_c3_binding_authority(binding,**args)
    except ValueError:
        pass
    else:
        raise AssertionError(key)
reference=weakref.ref(binding)
del binding
gc.collect()
assert reference() is None and identity not in production._CUDA_ARENA_C3_BINDING_AUTHORITIES
print('shared registry exact tuple and weak-reference cleanup verified (CPU bookkeeping only)')
