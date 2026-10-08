# Native-scope preflight repair

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note and fresh code/native preflight request
Author: codex
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: native-scope-aggregate implementation-note/code-preflight-review v1;
native-scope-cache design-request/design-review v1; native-scope-aggregate cpu-test-evidence v1
Owner Summary: Address the two independent preflight findings without changing
production code or permitting new GPU execution before independent re-review.

Base b0f22fb4f6eda395f72ae34f37044ed918254184; packet supplies the exact revised
HEAD. Full original Request changes report is archived unchanged at f630f81,
SHAb18527c80bc1daa951879de99ed39aeed4c8942685635e5e39e48a7daf3adeca.
The bounded supplement at083688ee95bdef9c7386a41f9828b0bff7cb5c16,
SHA393a32720ce10ddc525f132c38e5109a45b537a8a5df4ffa08524013e078a68c,
was independently approved for design only; report SHA
c0223ca5bdc303e2ff94c8f1bd50ac785cf50ded455393171a9cb321155f43f7,
archived unchanged in8b655a2. The governing aggregate design v2 remains binding.

## Implemented delta

F1: fixed observer retains the already-loaded wp_get_error_string function and
verifies its existing []/c_char_p declaration. Before original stream creation,
raw target context/stream must be absent and the original error buffer exactly
empty. After the original held/journaled return and the existing injected-fault
point, raw target/stream identities and positive handles plus another exact-empty
read establish the observer-local cache prerequisite. Its failure marks UNKNOWN
before any potentially initializing native stream getter. No error is cleared,
no setter reruns and no additional GPU initialization is introduced. Every
poststream Warp getter requires the prerequisite; the first actual native stream
must equal the actual held returned handle before establishing the baseline.
Completed contradictory snapshots still append before failure; scope after
observations retain the direct context query, without extra error-buffer polls.

F2: each fixed C10 getter now records its name, arguments and primitive return
immediately, or holds and records the exception before propagating. Count,
every primary ordinal, logical device and explicit-target current stream use
one wrapper. Warp error bytes are recorded losslessly as hex; its native stream
getter uses the same wrapper. Existing CUresult/null handling is unchanged.
The collector/parser remain CPU-only; no schema/manifest/production repin is
needed because only the fixed test observer and CPU tests changed.

## Author checks and original counterexamples

Local Python3.14, plugins disabled, bytecode/cacheprovider disabled:

- Initial expanded observer suite:23passed2.33s.
- Final observer/parser/readback suite:78passed5.85s, no skips.
- git diff --check passes (ordinary LF/CRLF working-copy notices only).

ROBOT_SIM_WARP_CONTEXT_SOURCE points to the retained exact pinned context.py
text outside the candidate, SHAeb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee.
The new fixture parses its exact Device methods without importing Warp. Its
native core remains explicitly a CPU model. On the original void-setter error
branch the real revised observer rejects before the cache getter; the positive
model reaches it once. Stale/nonempty/NULL/wrong-type buffers, absent local proof
and zero/mismatching first stream are negative cases. The actual observer return
hook is exercised with original modeled return notification and already-held
owner, including its earlier injected failure. UNKNOWN, zero sync/completion
and no authorized release are asserted on those failures.

Ten actual-observer CPU query failures cover every primary ordinal and both
later logical/stream calls, asserting exact ordered preceding results plus
the failing getter/arguments, no later call and no complete row. Count failure
separately verifies retained exception/name/single invocation. Existing C1 wrong
context despite normal scope exit and CUresult/null counterexamples remain.

The original independent repro scripts/logs remain unchanged in c4_assets.
Their historical assertions describe b0f22fb's defects; the new tests exercise
those paths under the revised precondition and diagnostic contract. Neither
those CPU models nor the501-pass remote baseline creates new native evidence.
Exact revised-HEAD remote CPU checks will be supplied separately during frozen
review. No new native/GPU observer cases have executed.

## Requested checkpoint

Fresh independent code/native preflight must verify F1/F2 closure and inspect
the complete b0f22fb-to-candidate delta plus affected original observer/parser/
readback invariants. Verify actual retained function identity/ABI and source
proof, no missing getter guard, original call/retention/failure ordering and
partial diagnostic behavior. Approval here permits the designated16 actual
native cases only; it is not evidence that they passed.

All16 actual native-scope cases, required affected faults/source/persistence
regressions and whole C2d3 independent review remain outstanding. C2e/f,C4c3/c4,
C4d/B3,product GPU preparation,a3b,result/baseline,host,R2,shipping remain closed.
C4 continues and remains unaccepted.
