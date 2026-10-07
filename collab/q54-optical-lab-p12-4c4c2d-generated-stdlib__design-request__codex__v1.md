# C2d3: exact CPython-generated sysconfig data admission

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2; absent-resolution design v1; sysconfig gap-scan v1; native_bootstrap.py; runtime_environment.py
Owner Summary: Extend the existing trusted-interpreter branch for one exact CPython-generated CPU data module. Do not classify it as a third-party dependency, relax external naming/root rules, or enable a general generated-module prefix.

## 1. Gap and bounded change

Base89e89c15e1bcbca2ec1171d6814a9d4e9f478b83. The first supervised real Go2
attempt stopped at external event614, requesting
`_sysconfigdata__linux_x86_64-linux-gnu`. SciPy _lib/_testutils.py:44 calls
sysconfig.get_config_var('HOST_GNU_TYPE'); the pinned CPython3.12 sysconfig
imports this generated build-data module. It is not included in
sys.stdlib_module_names or sys.builtin_module_names.

On183, sys._stdlib_dir is
`/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12`; the generated71715-byte
file is directly beneath it, SHA256
07c6cf395105faefff17ea4a888e00858ca3da7e2f8f9266cb7953c772085683.
The actual sysconfig.py SHA256 is
e9a916680e27278b130d8c4add824d55b952bef284bee354083ca9b436d5103e.
These are inspected deployment facts, not independent execution approval.

The runtime schema deliberately permits executable file modules only under
site roots and requires identifier components; this generated name also has
hyphens. Do not weaken either constraint. Its actual stdlib directory is on
the frozen interpreter search path, so marking that directory as an external
site root would contradict the existing no-site-path contract.

This supplement extends only absent-resolution v1 section2's finite trusted
interpreter name set. The worker is still a trusted sealed process, not a
sandbox for malicious stdlib/native code. Generated sysconfig data has the same
immutable deployment trust as the interpreter's existing standard library.
No new third-party permission, resolver, user-supplied module name, or IPC field.

## 2. Immutable generated-interpreter descriptor

During the existing isolated CPU bootstrap, before installing its singleton
finder, freeze one internal descriptor for the supported CPython3.12 POSIX
interpreter. Its name is mechanically constructed only from immutable startup
interpreter facts, following the inspected default rule:

```text
_sysconfigdata_{sys.abiflags}_{sys.platform}_{sys.implementation._multiarch}
```

Freeze those exact intrinsic strings, sys._stdlib_dir, sys.executable and
sys.version_info with the existing topology identity. No value comes from an
environment variable, configuration/request field, package, observed failed
import, caller-supplied prefix, or execution receipt. Do not call
sysconfig.get_config_vars() or execute the generated module to discover its name.
This is a narrow CPython3.12 rule; other interpreter versions/platforms get no
generated-module permission from it. They retain ordinary stdlib delegation;
the existing external deployment version check still applies.

Require a nonempty absolute sys._stdlib_dir that resolves to itself, equals a
frozen cold search directory, and is not inside the repository or a site root.
The expected source is exactly `<stdlib_dir>/<derived_name>.py`, an ordinary
regular non-symlink file. Freeze its resolved path, source byte count and SHA256
before body execution, without importing it. Reject missing file, alternate
origin or directory/path ambiguity; do not search other roots for a replacement.
Hashing these bytes records the trusted deployment snapshot, not proof against
a privileged actor modifying the interpreter or stdlib during execution.

The descriptor is internal source-session topology state, not a public capability
or serialized declaration that can mint one. The module is not added to the
external environment schema/table. The existing interpreter executable/version
binding must still match at environment configuration. Historical environment
tables, including the reviewed Go2 table, remain unchanged.

## 3. Override rejection and exact resolution

Reject presence of `_PYTHON_SYSCONFIGDATA_NAME` at bootstrap, including an empty
string or the otherwise correct default name. Recheck absence at topology entry
and existing boundary audits. Never silently clear it, temporarily accept it,
or derive permission from it. The inspected3.12 sysconfig name-selection body
must remain the trusted deployment implementation; no helper monkeypatch.

The sole captured finder has one exact full-name match, separate from ordinary
stdlib top-level matching. Do not admit dotted descendants, lookalike names,
another ABI/platform variant, or arbitrary `_sysconfigdata_*` modules.
For that one name, resolve through the already frozen PathFinder binding using
only the exact frozen stdlib directory. Require a SourceFileLoader (exact type),
the exact name/origin/path, no package locations, no reload target, and unchanged
frozen source bytes. No Builtin/Frozen alternative or fallback to site roots.
Returning the standard spec authorizes only this exact CPU stdlib source.
Do not alter the installed standard file, external loader naming validator,
sys.path, sys.meta_path cardinality, path hooks or find_distributions behavior.

The generated module must not be preloaded before source custody; retain the
existing cold rejection instead of trusting an already supplied object. Its
actual first load occurs under the installed singleton finder. At boundaries,
if present, validate exact module/spec/loader origin and source snapshot; freeze
the observed object identity at its first successful boundary and require that
identity on later audits. Existing trusted CPython import caching is not a
hostile same-process capability boundary. An unexpected preloaded/replaced module
or changed descriptor/file fails closed rather than recreating a baseline.

Use one internal stdlib classification rule for finder dispatch and import-ledger
audits. That rule distinguishes ordinary frozen stdlib roots from this single
exact generated full name. Do not patch only finder dispatch and then leave
external audit to reject the successfully loaded module as an unlisted publication.
No external create/exec/publication event is fabricated for a trusted stdlib
import. Descriptor identity/digest is available to the eventual runtime-custody
evidence as descriptive interpreter provenance, never import authority.

## 4. Failure and custody invariants

All descriptor/environment/path/source drift uses the existing sticky failure
path. In an armed segment, the original controller/guardian retains failure and
exits through its existing quarantine policy; no COMPLETE, retry, second ARM,
fallback sync or ordinary cleanup is introduced. Pure CPU bootstrap rejection
remains before third-party execution. Standard-library error reporting after
poison is permitted only under unchanged frozen topology, as already designed;
it cannot authorize release or reopen external/repository imports.

No GPU API is called to construct or verify this descriptor. The change is
untimed setup only and does not affect O(1) acquire, event roles, or arena leases.

## 5. Required tests before whole C2d3 approval

- Cold -I -S -B subprocess: ordinary sysconfig.get_config_var succeeds on the
  pinned real interpreter; exact generated file/name recorded; singleton
  topology unchanged; no third-party import required for this local property.
- The real captured Go2 call reaches SciPy's original HOST_GNU_TYPE query, then
  both mandatory Go2 workloads must complete canonical materialization and
  actual guarded Warp/CUDA upload/readback. A CPU sysconfig success is not a
  substitute for that parent acceptance matrix.
- Present overrides: empty, correct default, arbitrary module and foreign ABI
  name all fail before generated body execution. Mutating the override after
  install fails before the next supported import/boundary.
- Wrong source path, symlink, byte change, missing file, wrong loader/spec,
  dotted child/lookalike/prefix requests, metadata drift and preloaded/replaced
  objects are rejected; assert no fallback resolver/body and no usable binding.
- Real external admission audits accept the legitimate generated module without
  external events; same-name replacement and byte drift fail rather than being
  hidden by a baseline skip. Repeated legitimate cached use remains valid.
- Preserve ordinary absent-query/import behavior, disabled-adapter tokens,
  creator publications, no site/plugin initialization, poison reporting, cold
  source checks and C0 A/K regressions. Windows/non3.12 CPU verification must not
  fabricate a Linux descriptor; no successful Go2 claim from those platforms.
- Repin exact bootstrap classes/source in the next policy/catalog generation.
  Historical releases remain immutable. Independently review changed code with
  the full parent checkpoint; no CPU subset is relabeled complete C2d3.

## 6. Authorization and exit

Independent design approval authorizes this bounded implementation only.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host-payload and shipping remain closed. No renderer or final readiness.
The selected worker retains its interpreter contract; losing-lane deployment
fixtures and temporary validation drivers follow the existing P12.4e cleanup
obligations. No shared collab protocol or private-agent instruction changes.
