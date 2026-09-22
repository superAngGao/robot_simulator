# Q54 P12.4 C4c1a — Physical storage binding follow-up

- Author: Codex
- Stage: C4c1a review follow-up for Claude
- Base: `5cb288ab9824e203ef2816f4ea57e1b9d3956fc2`
- Scope: the single remaining P1; C4c1b/c remain closed

## Fix

Prepared-input registration now freezes each a2 array's physical storage
signature, in addition to its Python object identity. The signature records:

- exact indexed CUDA device;
- storage base pointer and allocation identity;
- allocation byte capacity and active byte interval;
- runtime dtype, shape, and strides.

Torch uses `untyped_storage()` (including its storage identity and a strong
owner hold), `data_ptr()`, `numel()`, `element_size()`, device, dtype, shape,
and stride. Warp uses its device ordinal, pointer, capacity, dtype, shape,
and byte strides. Registration also checks the exact array-name set, shape,
and active byte count against the canonical a2 manifest.

`assert_cuda_prepared_authority()` and `assert_warp_prepared_authority()`
re-read the live signature and compare it to the frozen one. This happens at
untimed preflight; CUDA C3/C4 binding already calls the former before creating
provenance. No physical-signature traversal was added to measured `acquire()`.

## Negative evidence

- Real `CanonicalNativeInput -> prepare_cuda_native_input ->
  certify_native_readiness`: `tensor.set_(replacement)` keeps the same Python
  tensor but changes `data_ptr()`; authority validation rejects it.
- Real Warp preparation and readiness: replacing the same array object's
  pointer with another allocation's pointer is rejected; the original pointer
  is restored before the test's resource cleanup.
- The CUDA arena C3 adapter rejects a rebound a2 tensor before any sample
  acquire or enqueue; terminal event count stays zero.
- CPU unit tests cover the same-object pointer drift for both backends.

Focused CUDA arena/native/unit suite: 38 passed. Ruff on changed Python
files: passed. C4c0 and full non-slow regression results are recorded after
the clean-tree run.

## 关键思考

Comparing `is` on a prepared tensor proves only that the wrapper has not been
replaced. It says nothing about `set_()` on that wrapper. The fix stores the
allocation facts when a2 first registers the wrapper and validates them at
the existing pre-enqueue authority boundary. A CUDA graph/dataflow receipt
cannot safely repair this later by hashing labels, and measured acquire is
the wrong place for an O(number of arrays) scan.

The original Torch storage owner is retained by the authority registry until
the prepared wrapper dies. That makes the captured base pointer meaningful
even if a caller rebinds the wrapper, because the original allocation cannot
be freed and immediately reused at the same address during the comparison.

## Review request

Please verify the same-object storage mutation is now rejected for both
backends, and that the full physical check remains in untimed a2/C3/C4
preflight. This follow-up does not authorize C4c1b, C4c1c, product GPU
preparation, a3b, results, or baselines.
