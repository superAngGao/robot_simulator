# C2d3 design-round inspection and review evidence

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: inspection-evidence
Author: codex
Version: v1
Date: 2026-10-07
Status: accepted
Related Files: runtime-adoption designs v1/v2 and independent reviews v1/v2
Owner Summary: Evidence-only archive of design approval. No implementation,
runtime safety test, performance result or activation is claimed.

## Revisions and independent review

Base ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3. Initial design b54e82b received
Request changes from fresh session c4_c2d3_runtime_adoption_design_review_v1.
Two P2 findings concerned required optional missing imports and dependency-created
modules that never have individual file-loader histories. The unchanged report
SHA256 is32ae49cf06be44a2ca5d408269d96d28ad908798176d4c8999de22f64ffce729.

Revised design f668b5b3daa0124b02b474b423bce173e0dc1af4 was approved by a second
fresh session c4_c2d3_runtime_adoption_design_review_v2. Active design SHA256:
1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224.
Unchanged second report SHA256:
5357e814b9ce2d88cd699a60e583ce2badc72cad3b18ed7a6b4dbb0ecb86416c.
Both sessions used fork_turns=none. Candidate remained frozen during each review.
Reports contain their own commands, fresh observations, limits and process checks.

Only bounded C2d3 design implementation is authorized: exact environment/source
admission, dependency custody, runtime binding, shared a2 upload and supervised
validation/readback/disposal in fixed sacrificial drivers. All later/product gates
remain closed. Actual cold imports, dual-GPU failures, ownership tests and the
complete environment inventory are implementation acceptance work, not passed here.

## Author source inspection

Read project instructions and approved parent/source-custody designs; inspected
native_bootstrap/native_setup/guarded_lifecycle/native_contract/native_input and
real materializer calls. An attempted native_materialization.py read failed because
that filename does not exist; rg located native_input.py/render_parameters.py.
This read error was not counted as a product defect or verification result.

Read installed sources on primary tileops-dev, hostname hd03-gpu2-0031, under
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages. Observed
metadata: Torch2.9.1+cu128, Warp1.12.0, NumPy2.5.1; reviewers additionally checked
trimesh5.0.0 and absent optional specs without importing packages. No environment
install/change and no backend initialization/test occurred in this round.

Source SHA256s observed at20:04:50 server process timestamp:

- torch/__init__.py: 3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288
- torch/cuda/__init__.py: 9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99
- torch/cuda/streams.py: cdad950977c5eb5a0fe744eaff8c7f9194817de40f380b40fdbc29032bad3fef
- warp/_src/context.py: eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee
- warp/native/warp.cu: 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b

These are selected source observations, not the full frozen deployment manifest.
Verified facts include Torch plugin auto-loading/module publication, Warp greeting
memory queries and native primary-context retention, deferred stream/context setup,
and trimesh optional probes. No source fact was represented as a real GPU test.

## Remote snapshot and transport correction

Created a new detached worktree only:
/data2/shared/ang.gao/robot_sim_c2d3_design_v2 at exact f668b5b, clean, matching
design digest and git show --check. Prior code worktrees were preserved.
Original /data2/shared/ang.gao/robot_simulator remains clean mainf771e61.

The first incremental git bundle command used an unnamed positive commit OID
and returned 'Refusing to create empty bundle'. Subsequent file/scp/fetch/worktree
attempts consequently failed; no target worktree or valid transfer existed.
Corrected command used 0dd6848..HEAD, then checked each subsequent step's exit.
Local/remote bundle SHA256 both:
5bcb669dafd75a05077f8d72161bac716b25bfcec5ca77b3d85f0c7065f6580a.
Fetch and new worktree creation succeeded; remote HEAD/digest were independently
rechecked by reviewer v2. No push, reset, merge or original-tree edit occurred.

## Round-end process hygiene

Author launched only foreground Git/read/SSH/SCP commands, no Python/GPU test,
worker or supervisor. Completed commands returned; short-lived utility PIDs were
not all sampled, so this is not a complete process ledger or supervisor-reap claim.

At20:21:43 the primary ang.gao process list showed no Python/test/worker/supervisor,
only user-session and current SSH inspection processes. Observed shell3127192,
parent3127191; subsequent exact ps lookup showed both absent (header-only ps exit1
is the expected no-matching-process result). Local Win32_Process lookup returned
no ssh.exe/python.exe/python3.exe residual.

GPU inventory included pre-existing MPS2238410 and unrelated xdist processes
3115355/3115362/3115372/3115375/3115386/3115383/3115393. Exact ps attributed them
to UID232071, parent3115267, started20:19:54-55; current ang.gao UID1006. None
were launched or touched here. No global-idle claim, unrelated signal, GPU reset
or diagnosis of the historical101hang is made. No owned unresolved hang observed.

Documentation diff/show checks pass. The final archive changes only this evidence,
the unchanged second review and the scoped PROGRESS entry. Approved design/code,
shared workflow, gate.json and product activation remain unchanged.
