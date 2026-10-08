# Torch publication: callback-free failure intake

Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-10-08
Status: in_review
Related Files: approved design-request/design-review v1; implementation-gap-review v1/v2; implementation-note v2
Owner Summary: Repair F4's callback-triggered loss of sticky failure. Request a fresh independent bounded review; actual captured Torch, aggregate custody and whole C2d3 remain unfinished.

Code predecessor:68e10d2b13249dfb38d6a92f4335e344602f4854. Evidence-only base:
cbfb1ce (unchanged complete independent report v2, SHA-256
b273f1b08d265f94d22d1ae0cb813a2a26eb55a3a27be2da69cc1968c059c68c).
Candidate is the commit containing this note. The original v2 repro remains
unchanged outside the repository, SHA-256
e30aedcdbaa19a56a6c4a47c6a1127d84ebd151b9354e337fdde4e6d2c78721a.

Alias custody intake retains every available publisher first, then traverses
metadata only for exact ModuleType. It no longer calls isinstance on the
incoming value, avoiding __class__ and __getattribute__ traps. The alias
admission contract already requires exact ModuleType; this adds no permission.
SourceSession.fail commits failed state and original-error identity before
ancillary intake, while still retaining the available graph before forwarding
to external failure and the first controller termination. No second terminal
operation or post-poison successful event is introduced.

Cold source-session tests cover both traps through audit, direct fail and
already-poisoned intake. They assert zero callback calls, retained publisher and
original error, no successful alias fact, and no audit recovery after removal.
Actual external/source failure routing with a CPU controller endpoint asserts
both flags, full available alias/replacement custody and the original exception
before the first termination; late invalid publishers are retained with no
second termination. This is not native quarantine/reap evidence.

Local Python3.14 executed after source-policy/catalog repinning:

```text
pytest test_interpreter_alias.py test_torch_module_replacement.py -q -p no:cacheprovider
77 passed in82.08s
pytest test_runtime_import_admission.py test_native_global_publication.py -q -p no:cacheprovider
214 passed,3 skipped in5.81s
```

Paths above are under tests/unit/optics. Skips are pinned CPython3.12/POSIX
generated-stdlib variants on Windows3.14. The two unchanged external repro
files were rerun together:6failed,1passed in12.78s. Four v1 old-acceptance
assertions reject as before; v2 audit no longer invokes the expected callback,
and direct fail returns with poison instead of raising that callback. Their
failure output is counterexample rejection, not six passing tests. New explicit
contract tests above assert the repaired outcomes. Raw combined log:
../c4_assets/torch_gap_v3_original_repros.log. All local test commands exited.
An initial staging command used a nonexistent catalog path, then was corrected
to optical_worker/data/native_sources/v11.json; this did not affect tests/pins.

Predecessor68e10d2 primary183 CPU results were retrieved, not relabeled as this
candidate:373passed in164.28s,no skips, eight modules listed in notev2. Log
c2d3-68e10d2-torch-cpu-v2.log SHA-256
50e106949a45a6242b8b88c075b8e89a863d65ade207067764fcebddf0cea310.

Outside-candidate ordinary deferred-init diagnostic at predecessor68e10d2:
/data2/shared/ang.gao/c2d3-68e10d2-torch-deferred-init-diagnostic-v3.
Parent161535,supervisor161540,worker161541; worker74,reaped; supervisor/parent0.
Scoped ps confirmed these plus invocation parent161534 absent after exit.
Original ACK1000ms/operation120000ms;1ARM,0normalCOMPLETE,0sync,no normal
release,UNKNOWN throughout. It imported ordinary Torch and created a stream
on cuda:1 before an intentional exception/hard exit. Raw native observations:
ambient0 before/after; primary contexts before all8false, after only device1
true; returned stream.device.index1. Observations SHA-256
8455afac78c69f5432a8447cf863dd2cdc2a75d0e7c9b3ae7831e842b94e4411.
956modules match prior ordinary import census. One additional mapped path:
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.71.05, requiring a fixed profile
pin before captured deferred initialization. Pre-existing MPS2238410 remained
untouched. This is ordinary diagnostic evidence only: no captured admission,
runtime binding, a2 upload, success COMPLETE or full native scope proof.

The drafted captured diagnostic has not run. Its UNKNOWN-before-import hook
would correctly stop normal production importing immediately, so it requires
further scope/design inspection before execution; do not report it as executed.
All previous exclusions remain, including aggregate schema and both-lane
canonical/replay/real-Go2/failure/hang/dual-GPU evidence. Later gates remain
closed, and C4 remains active.
