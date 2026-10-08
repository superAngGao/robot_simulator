# C2d3 author deployment fixtures (not product profiles)

`183_cuda_cold_import_author_v1.json` freezes the bounded Torch cold-import
candidate (canonical SHA256
`2ffb9a81595b60855db8768f2892409c49d32d8ec97d41a3673b18e456492e07`).
It covers857 files,758 file modules,42 publications,7 replacements and8 exact
absent probes. Actual captured import on production `a3d750b` reached the fixed
pre-Stream checkpoint, then deliberately quarantined under the existing guard.
See torch-captured-import test-evidence v1 for source provenance and every failed
draft. This candidate is under independent review and does not freeze full CUDA
initialization, uploads, Go2 or native-loading closure. In particular deferred
NVML loading is not yet included. No product launch selects this fixture.

`183_warp_synthetic_author_v5.json` is the active schema-v5 candidate for cold Warp
asymmetric/dynamic canonical-input upload/readback on primary183 (H200, robot_sim
CPython3.12.13). It grants no worker IPC/product activation. The opt-in test must
run without skips on the matching host before its narrow result can be cited.
This is NOT the complete required Warp/CUDA/Go2/multi-device acceptance matrix.

Canonical contract SHA-256:
`0783cc820dcb738b8ec82586e2d14a12a7b9b81989e67d1ba9f86ce1ea3ed1b7`.
The hash covers canonical encoding, not the file's raw bytes.

The required `native_globals` and `disabled_optional_modules` tables are empty: synthetic
inputs do not import trimesh. No dependency permission was added during this
schema migration. The publication table also remains empty: these workloads
do not need the newly specified native creator-self aliases. Versioned Go2
profiles and their complete closure still require runtime validation and review.
The active Go2 candidate is `183_warp_go2_author_v3.json`, canonical digest
`88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8`.
It adds only the three independently design-reviewed native-global interval
rows to the prior file/publication/absence inventory. Actual captured admission,
upload/readback and full C2d3 acceptance remain required. No discovery adds rules.
Synthetic v3 (canonical digest
`077399dc319dd9bb2d0ace5f6f776743c6b6b50b8bc14a410a87d68438eb4903`)
and Go2 v1 are preserved as historical v3-schema artifacts.
`183_warp_synthetic_author_v2.json` is retained unchanged (canonical hash
`f42485bdf9580d56b86080aad18d2d3e5a8832c23bd0868419852b2da4b734af`).
`183_warp_synthetic_author_v1.json` is retained unchanged as
historical evidence (canonical hash
`ae95257eb95b4441e5ae786a5ae65f3e7a86459e30aa455abfc27ecae0ed2299`).
The v5 parser rejects earlier schemas; it does not silently migrate an old contract.
Synthetic v4 and Go2 v2 remain unchanged historical schema-v4 artifacts.
Both new Warp profiles add only the empty `module_replacements` table. Their
dependency permissions are unchanged. The seven Torch replacement rows belong
only to the independently approved Torch supplement and its future CUDA profile.

Authoring source: installed-file inventory
`/data2/shared/ang.gao/c2d3-offline-wheel-inventory-v1.json`, SHA-256
`5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a`.
Its 32-wheel inventory was NOT granted blanket module permission. The candidate
has 86 named NumPy materialization modules and 25 named Warp runtime modules.
They derive from inspected eager imports, the native multiarray dependency names,
and the actual `numpy.typing` materializer dependency. The sole absent probe is
`numpy._distributor_init_local` requested by `numpy._distributor_init`.
No runtime trace adds an entry automatically. No namespace/creator exception is
needed for this observed Warp path. Torch and optional trimesh/Go2 dependencies
are not admitted by this fixture.

Files include the named Python/extensions, NumPy's three bundled native libraries,
Warp's two native libraries, distribution METADATA, inspected Warp native source
and the explicitly observed OS linker dependencies. Driver libcuda595.71.05 has
a separate `os_native` root entry; it is NOT claimed to come from a wheel.
Read-only `ldd`/resolved-file hashes supplied these OS facts. Full native-loading
closure, per-device context observations and failure coverage remain review
obligations; successful input parity alone does not prove them.

The fixed test uses `ROBOT_SIM_TEST_C2D3_COLD_WARP=1`, actual captured source and
real supervisor in fresh `-I -S -B` children. Operational ACK/operation budgets
are watchdog bounds, not performance gates. Input readback is validation, not
B3 delivery. The end is one supervised target sync/release/hard exit, not final
optical readiness or a benchmark result. Sync failure/hang retain the failure
semantics and require real child reap, never an outer pytest timeout as success.

Before approval of C2d3, freeze the remaining environments/evidence schema and
complete every required real-device row. Do not treat this fixture or its tests
as an independently approved CPU/GPU sub-checkpoint. P12.4e removes losing-lane
and temporary fixtures only after equivalent selected-worker failure coverage;
the cleanup ledger retains this obligation.
