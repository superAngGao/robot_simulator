# C2d3: no direct Python-visible owner relation in the installed native modules

Initiative: q54-optical-lab-p12-4c4c2d-cython-owner-probe
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: draft
Related Files: cython-publication test-evidence v1;
optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py
Owner Summary: A bounded ordinary CPU probe rules out claiming a direct
Python attribute or GC referent relation for the inspected installed extensions.
It does not prove absence of native ownership, authorize a new publication
kind or replace independent review.

Both local and primary183 isolated worktrees were clean at
6958f788e8fc2776b47688878e3ab8ef985daf58 before this probe.
No tracked implementation changes, package installs or GPU preparation.

## Exact diagnostic

Executed via stdin to SSH tileops-dev:
`timeout 30 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`.
Exit0, no timeout. This is outside captured worker custody and is not admission
or GPU evidence. It imports installed CPU extensions in a short-lived process,
not a substitute canonical materializer. Assertion at end excludes Torch/Warp
imports. No site.main(), environment change or persistent Python process.

```python
import sys, importlib, gc, json
sys.path.insert(0, '/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages')
for name in ('numpy.random._bounded_integers', 'numpy.random.bit_generator',
             'scipy._lib._ccallback_c'):
    m = importlib.import_module(name)
    refs = gc.get_referents(m)
    rows = []
    for k in ('cython_runtime', '_cython_3_2_5', '_cython_3_2_8'):
        child = sys.modules.get(k)
        if child is not None:
            rows.append({
                'key': k,
                'raw_owner_attributes': [a for a, v in vars(m).items() if v is child],
                'direct_gc_referent': any(r is child for r in refs),
                'refcount': sys.getrefcount(child),
            })
    print(json.dumps({'native': name, 'file': m.__file__,
                      'direct_referent_count': len(refs), 'observations': rows}))
assert 'torch' not in sys.modules and 'warp' not in sys.modules
```

## Observations and limits

All three extensions resolved under the named robot_sim site-packages to
their corresponding cpython-312-x86_64-linux-gnu.so files. Each returned one
direct GC referent. Every observed pair had an empty raw_owner_attributes list
and direct_gc_referent=false:

| Native module | Present queried Cython keys |
| --- | --- |
| numpy.random._bounded_integers | cython_runtime, _cython_3_2_8 |
| numpy.random.bit_generator | cython_runtime, _cython_3_2_8 |
| scipy._lib._ccallback_c | cython_runtime, _cython_3_2_5, _cython_3_2_8 |

Refcounts were12/6 for each NumPy observation, and14/5/6 after SciPy import.
They include diagnostic references and say nothing about ownership provenance;
they are not proposed gate values. Empty direct-reference observations do not
exclude a deeper attribute graph or opaque native static/module-state ownership.
No such relation has yet been mechanically demonstrated. Do not fabricate one
from a version-like name, matching metadata or total reference count.

Next action remains an independent bounded gap review deciding the minimal
honest publication/ownership contract, followed by a design review before
implementing a capability change. Strong failure-graph retention from the real
captured attempt remains a separate, demonstrated fact.

Fresh fork_turns=none review creation was retried as
/root/c4_c2d3_cython_publication_gap_review_v2 and again failed with
"agent thread limit reached". No agent ran and no review verdict exists.
This is the second consecutive turn encountering that same review-capacity
limitation; the first archived the captured failure, and this turn added the
direct-owner counter-evidence. Neither turn grants independent approval.

C2d3 remains unfinished; all later-stage/product/a3b/result/baseline/host and
shipping gates stay closed. Shared collaboration rules and frozen deployment
tables are unchanged.
