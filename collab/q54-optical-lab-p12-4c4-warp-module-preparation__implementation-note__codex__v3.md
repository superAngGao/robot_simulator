# Warp preparation: actual IDLE entry repair

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: code-review repair before native acceptance
Author: Codex
Version: v3
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation code-review v2; test-evidence v1; design-request v2
Owner Summary: Moved the pure CPU healthy-IDLE/scope check ahead of resource
  validation so unknown scope cannot invent a failed normal guard. Preserved
  existing lifecycle policy and added actual native fault wiring. Fresh review
  and the complete native acceptance matrix remain required.

Base0194cb2442c52300b06fd33f7042ed291e723af3. Fresh independent reviewer
/root/warp_module_code_v2 closed F1 at the CPU code boundary and requested F2.
The full report is archived unchanged as code-review v2, SHA256
ce8b82ac6aeede707fee6fb7995a29b23afd7e1da591f23a27df1ff7be960617.
Its four regression suites completed158 passed,9 skipped in269.80s; its4
original ABI repair checks and4 compile hash-transition checks passed.

F2 trigger: input_validated, genuine IDLE, valid report and original runtime
binding, but sticky unknown scope. Resource validation called require_adopted;
that public lifecycle API rejected compromised scope by marking a guard failed,
even though none was active. The existing unknown test invalidated the report
first and missed this path. Both author and independent reviewer reproduced a
guard_failed event with guard:null; quarantine/no-sync already held.

The preparation entry now burns its existing single-use flag and checks IDLE,
no active context/reference, and uncompromised scope before any adopted-resource
API. Its ordinary CPU rejection reaches the existing IDLE failure handler. No
controller policy, new state machine, synchronization or guard is introduced.
The permanent regression now includes a valid-report unknown-scope case. An
active-guard control ensures that rejection inside a real existing segment still
uses that original guard, never the IDLE disposal path.

Author verification after the repair: session88650 exited0 with34 passed in8.38s
(24 module-preparation cases, independent F2 valid/invalid-report repro2, prior
F1 adapted repro4 and compile hash-transition cases4). The subsequently added
active-guard control passed with the full25 module tests (5.23s, exit0).
A fresh continuation reran those25 and the independent two-case F2 repro:
27 passed in5.84s, exit0, using Python3.14 -B and no pytest cache. Native support now
includes a fixed unknown-entry case with an otherwise valid real Go2 report;
its runner requires6 completions/6 arms, worker74, zero synchronization and no
guard_failed event. That new case has not executed at this record cut.

Additional author native evidence at predecessor0194cb2:

- /data2/shared/ang.gao/c4-warp-module-author-v4: Go2 960x640 succeeds with the
  repaired ABI snapshot; tool87788 exit0, worker2803385 exit0/supervisor2803383.
  Three outer compile snapshots record exact before/after equality, ambient
  device0/current-context-none restored, separately from inner target1/stream
  observations. Post-exit artifact checks verify9 hashes and absent owned PIDs.
- v5: actual compiled-kernel dtype mutation rejects at the real validator with
  ValueError("Warp executable or kernel ABI changed"); tool26149 exit0 as a
  negative acceptance check, worker2812987 exit74/supervisor2812979.
- v6: actual compiled-kernel ndim mutation rejects at the same boundary;
  tool31144 exit0, worker2855443 exit74/supervisor2855440.
- v7 entry-hash mutation completed in tool35549, exit0; worker2913574 exit74,
  supervisor2913570, parent2912530. The expected negative case passed at0194cb2.
  The remote checkout remained unchanged through terminal completion.

These cases do not close remaining native load/publication/adoption/hang,
source/cache/options/artifact/record, actual-IDLE issuance and failure restoration
coverage. No optical output, warmup, final readiness or performance is claimed.
Continue the complete parent graph/timeline/arena, rendering/lifecycle, C4c3/c4,
C4d/B3, systematic refactor and fresh whole review. Product/result/baseline and
backend-selection gates remain closed.
