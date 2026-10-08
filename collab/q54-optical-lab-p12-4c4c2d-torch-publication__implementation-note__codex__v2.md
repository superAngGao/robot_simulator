# Torch publication: repairs for independent F1-F3

Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: implementation-gap-review v1; approved design-request/design-review v1; implementation-note v1
Owner Summary: Repair the three reproduced custody/alias defects, with earliest-terminal regression checks. Request a fresh bounded implementation gap review; actual captured Torch and the final C2d3 aggregate/matrix are still required.

Reviewed predecessor:64db378dfc92996635ebb5aa644272e080e1bf80. Its complete report
is archived unchanged in commit7ee9ef0, SHA-256
ac6fbf54c2206721ceffbb9b718b48c2aab3b3a82e224ca421fe910f1ff037aa.
Original bad-acceptance repros remain untouched at
`../c4_assets/test_torch_publication_gap_v1.py`, SHA-256
57bbfead7bae23deab212907e020e3ab2b9be935421c77052b783b669b79e63a.
This candidate is the commit containing this note; its base is7ee9ef0.

F1: SourceSession now has one intake-only boundary operation that reaches the
registry-held admission, retaining finite replacement/native-global values and
the alias graph before validations, frame checks or terminal handling. Entry,
return, both audits, alias observation and SourceSession.fail use it. It neither
adopts normally nor emits facts. SourceSession failure also routes through the
existing monotonic ExternalAdmission failure path before the first termination.
Tests use the real alias observer and SourceSession.fail with a controller
endpoint that records the graph at its first terminate entry. Request, return,
malformed wrapper, malformed frame, source audit and native-global cases all
retain the available value before that endpoint. They assert no new successful
facts after failure; this CPU endpoint is not a real quarantine/reap proof.

F2: One strict fact-projection comparator checks exact dictionary/key/value
types and values. Native-global/replacement/alias event checks share it;
replacement token projections also use it. Equal bool/int/float substitutions
reject in healthy and poisoned issuance. The CPU core does not import a
later-phase schema merely to validate its own held facts.

F3: An issued multiprocessing resolution permits the legitimate non-final
pre-publication interval, but final audit rejects a missing/None publisher.
Tests cover removed, None, initializing-then-removed and resolution-only states.
No-resolution optional absence and normal exactly-once observation stay legal.

After repinning AST policy v12/catalog v11, local Python3.14:

```text
python -m pytest tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_native_global_publication.py -q
281 passed,3 skipped in75.22s
```

The original four bad-acceptance repros were rerun unchanged: all four now fail
their bad-behavior assertions (7.60s). Both F1 probes see retained=True at first
termination, while F2 raises interpreter-alias facts drift and F3 raises publisher
disappeared. This is rejection evidence, not a relabeled passing test suite.
`git diff --check` passed. Local author pytest processes all exited.

Matching-primary183 predecessor64db378 CPU validation completed:
349 passed in144.80s, no skips, command/SSH exit0. Modules: alias,replacement,
native_runtime_driver_process,runtime_environment,runtime_import_admission,
native_global_publication,runtime_import_cold,generated_stdlib. Raw log
`../c4_assets/c2d3-64db378-torch-cpu-v1.log`, SHA-256
8d9ff552f96cce6299be20588cb187dbd3b255e836f224774f09f69d4d7b5d4c.
This predecessor evidence is not reattributed to the repairs.

Parallel read-only work outside the frozen candidate produced a CUDA profile
draft (758 modules,42 publications,7 replacements; canonical digest
dda212931b091cb0dc5f3f55f868cbebde7cf3f9594f838a44bf7dddec43f8c7).
It is not installed, runtime-complete or admission approval. Optional-source
audit `c2d3-torch-optional-probes-audit-v1.json` SHA-256
0c2c95bd79610d45852143f547960faef38b28999e891df39ee14cf239d3a6ca;
its PID143844/parent143842 exited and scoped ps found neither.
Native static audit `c2d3-torch-init-native-audit-v1.json` SHA-256
1b01df2dc18acde8ec7d7939a6c6daf8c9f901576eac6626ea79944965be18ec;
PID156585/parent156582 exited and scoped ps found neither. Hash-verified installed
libtorch_cuda.so contains a MagmaInitializer callback consisting of `ret`; the
actual magma_init symbol is in the separate linalg library. This agrees with the
lazy branch in the [exact upstream source](https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cuda/LinearAlgebraStubs.cpp).
It narrows one scope-audit question, not the complete deferred-init/exception
safety or target1/ambient0 proof. Neither audit loaded Torch/CUDA.

All implementation-note v1 exclusions remain: real captured Torch, complete
native scope/deployment closure, cross-guard failed-prefix aggregate replay,
both-lane canonical/replay/real-Go2 upload/readback and full fault/hang/dual-GPU
matrix plus independent whole-stage approval. No later gate opens; C4 is active.
