# C2c5 bounded repair of independent review F1-F3

Initiative: q54-optical-lab-p12-4c4c2c-source-policy
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-03
Status: in_review
Related Files: source-policy implementation-note/review codex v1;
optical_worker/native_source_policy.py; tests/unit/optics/test_native_source_policy.py
Owner Summary: Fix deployment-interpreter portability and explicit forbidden
alias/reflection spellings. Same read-only primitive scope; no worker admission.

Original base7f32e68; previous candidate3a365ea. The complete independent v1
report is archived unchanged. Its Request changes and failed commands remain
historical evidence, not overwritten by this repair. Candidate is the commit
containing this note, identified separately in the next review packet.

F1: AST locates the single top-level snapshot function, but no interpreter AST
serialization enters its digest. Pin its exact physical UTF-8 source lines,
including the final newline, to
`725c7f315fdba4ad6df78e31a149865fd97e26da3c95df24678dbadba0459199`.
The full renderer release remains pinned to
`636019f65594ac83dce1b386f4b3f684053e419a8cf81ba5feddce237cf318e4`.
No newline normalization or release-hash fallback is accepted. Exact POSIX
file validation is now explicitly tested alongside byte validation; Windows
CRLF uses a declared LF release fixture, not a false checkout-identity claim.

F2: reject `from builtins import eval/exec/compile/__import__`, with or without
aliases, and wildcard imports. Permitted builtin imports remain valid.

F3: reject explicit forbidden name/attribute string literals in call arguments,
keyword values and subscripts regardless of direct, qualified or imported-alias
call spelling. This conservative **restricted source dialect** also rejects a
harmless call carrying exactly such a forbidden string. It is not arbitrary
Python dataflow analysis or protection from hostile in-process manipulation.
Inert policy data containers and ordinary reflection literals still pass.
The frozen policy digest includes these exact rules.

New negatives cover the independent reviewer repros, all four reflector names
and four forbidden attributes across direct/qualified/imported/module aliases,
keyword/subscript lookup, and both alias and unaliased forbidden imports.

Validation records before final freeze: the first repaired collection passed
176 tests locally; one invalid Python-syntax parameter was then removed rather
than counted as a test that merely returns, leaving175 meaningful cases. Final
Python3.14/Python3.12 commands and actual counts belong to subsequent evidence.
The original deployed3a365ea broader source/protocol/core collection had105pass
and2fail (F1); existing contracts/render migration separately passed87 on remote.
The latter local collection had62pass/25missing-Torch failures and is not claimed
as passing locally. No installation, fallback policy, GPU test or result export.

All scope exclusions and cleanup obligations in v1 remain. Source catalogs,
bootstrap/import ledger, C2d/e/f, C4, GPU preparation, B3, shipping and a3b/result/
baseline gates remain unimplemented or closed as applicable. Old C0 is unchanged.
