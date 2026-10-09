# Pinned setuptools initialization inside CUDA compilation custody

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: bounded design amendment
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; setuptools-admission gap-scan/gap-review/test-evidence v1; systematic-refactor implementation-note v1
Owner Summary: Admit the installed setuptools83.0.0 dependency through one source-session-owned initialization transaction. Keep one custodian finder and the existing same-process load_inline/direct-load/guardian boundaries. Explicitly support its finite vendor imports, one Python package alias, one required resource and its own version metadata. No general packaging, plugin discovery or arbitrary import adapter is introduced.

## 1. Scope, evidence and change in authority

Base code/evidence HEAD: `4ac1e0dd0b42a08c9490152e84d47969e3d73da0`.
Primary deployment remains tileops-dev, robot_sim Python3.12.13, Torch2.9.1+cu128,
Conda setuptools83.0.0 build py312h06a4308_0. Previous approved compilation design
v3 SHA256 is `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
Its guards6/7/8, terminal9, recipes, generated-loader boundary, native observations,
job ownership, immutable input cut, deadlines and higher closed gates are unchanged.

This amends its unchanged external-import/publication assumptions. It authorizes
neither unobserved dependency import nor generic finder delegation. In particular,
the two setuptools shim functions will be interposed temporarily; this is an
explicit adaptation, not a claim that all dependency initialization is unmodified.
Installed source files, original spec producer and its actual custom loader still
execute from inventoried bytes. No source rewriting, dependency install, .pth
execution, preimport outside custody, build subprocess redesign or prebuilt module
substitution is included.

Evidence is the unchanged independent gap review at758950c and its test-evidence
v1. Helper archive SHA2568f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3;
target importlib/data archive SHA2563a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289.
Actual target CPU stand-ins established the alias metadata transitions and missing
resource/metadata support. They did not execute setuptools or prove a complete
transitive closure. Complete explicit source-audited rows are an implementation
preflight deliverable, frozen before any original input phase; runtime observations
must never append permissions.

## 2. One owner and one rule source

Add private captured module `optical_worker.setuptools_admission` in the CUDA
continuation closure, holding `_SetuptoolsInitialization` and its fixed helper
objects. It is issued only by the actual source session for that session's active
compilation root. Private identity registration binds session, external admission,
environment, controller, compilation root, native thread and first setup guard6.
Copying its fields, constructing another instance or replaying a descriptor cannot
activate it. Retain the object in the existing compilation root before registration,
adoption or any state transition can fail. Adopt it as HOST_OWNER, role and operation
`setuptools_initialization`. Ordinary file modules continue using existing external
loader, file checks, import-call journal and external_import_owner adoption.

Bootstrap delegates only fixed dependency-specific decisions to this owner: the
one topology extension, exact shim interval, exact alias root, named metadata and
declared resource. There is no adapter registry or user-supplied callback. Shared
file verification and sticky source/controller failure remain in their existing
owners. The new module must not duplicate a file importer, token registry, guardian,
destructive cleanup stack or distribution scanner.

Extend runtime-environment to v6, preserving v5 interpretation unchanged. v6 adds
exact field `setuptools_initialization`, either null or this exact object:

```
{"policy":"setuptools_83_conda_cuda_g1_v1",
 "vendor_root":"setuptools_vendor",
 "source_package":"setuptools._distutils",
 "alias":"distutils",
 "metadata":{"root":"site","path":"setuptools-83.0.0-py3.12.egg-info/PKG-INFO"},
 "resource":{"package":"jaraco.text","root":"setuptools_vendor",
             "path":"jaraco/text/Lorem ipsum.txt"}}
```

All keys/values above are literal policy, not selectors for another package.
Actual bytes/sizes/hashes live only in existing files/distributions rows. Require
setuptools distribution83.0.0, exact source-file layout from the pinned archive,
and vendor root equal to resolved `<site>/setuptools/_vendor`. No duplicate physical
file rows through both roots. The normal explicit module, namespace and absent-probe
tables describe every other permitted name, phase, origin and requester. Distutils
root is reserved by this descriptor, cannot collide with modules/publications/
namespaces/replacements/native-globals, and may parent declared distutils.* file
modules. Such children are individual requested-name file executions, never implied
aliases of similarly named setuptools._distutils.* modules. No prefix permission.

For every top-level external name, resolution order is site then setuptools_vendor;
the order is fixed by this policy, not root-ID sorting. Legacy v5 ordering remains
unchanged. The vendor root is available to explicit external resolution only after
transaction begin; rules for vendor and distutils names cannot execute in the input
prefix or other lanes. Earlier loaded site packages (e.g. packaging) retain their
existing object/origin; vendor rows cannot replace them. Freeze actual namespace
locations and backports' finite pkgutil.extend_path result in the ordinary rules.

## 3. Entry and topology

Begin inside the first actual compile device/stream context, after guard6/root
registration/toolchain inventory, before importing torch.utils.cpp_extension.
Use the existing observed outer cuda_compile leaf; add no native leaf or extra
setup epoch for this CPU transaction. The same session cannot begin twice.

Entry requires no setuptools, _distutils_hack, distutils or backports package or
subtree in sys.modules or prior records; no foreign finder or path mutation; no
prior dependency transaction. Previously admitted Torch/input owners remain held.
Freeze SETUPTOOLS_USE_DISTUTILS=`local` in the trusted worker environment and audit
it with the existing toolchain environment contract. Use the existing fresh0700
run root as worker cwd before bootstrap; keep its identity/cwd unchanged and
`pybuilddir.txt` absent, including dangling links, through spec production. Thus
the source's is_cpython branch is false. Do not invoke a .pth startup shim.

The session makes one transition from original path tuple P to
P + (exact_vendor_path,), recording both tuples. No site root is added to sys.path.
Preserve the sys.path list object; require vendor absent before transition and
exactly once afterward. Update only the session-owned expected topology at the
same synchronous boundary. The pinned setuptools sys.path.extend then does
nothing. Its backports pop is likewise a verified no-op because absence was
required; reject preexisting owners instead of deleting them.

sys.meta_path remains the identical singleton list containing the identical
custodian throughout. Standard-library top-level PathFinder resolution uses P,
not the appended vendor directory; builtin/frozen resolution stays unchanged.
Stdlib child resolution uses the admitted interpreter parent's standard path.
No vendor file may shadow a trusted interpreter module. Include generated
sysconfig and terminal-after-poison imports in these tests. sys.path_hooks and
standard resolver method identities remain frozen. On failure do not restore P
and pretend initialization never happened: preserve owned state, poison, and
allow only existing terminal diagnostics/drain logic. Successful P+vendor remains
fixed for guards7/8 and terminal9.

## 4. Exact shim integration and state transitions

Successful state path:

`NEW -> ENTERED -> HOOKED -> SHIM_ACTIVE -> ALIAS_CREATED -> ALIAS_EXECUTED -> ALIAS_PUBLISHED -> SHIM_CLOSED -> RESTORED -> COMPLETE`.

An alias spec is retained in SHIM_ACTIVE before create begins. SOURCE package
loading occurs inside that interval through the ordinary source loader. Resource
and metadata reads are side events after ENTERED and before COMPLETE; no new state
cycle. Any fault latches FAILED permanently; restoration attempts have their own
once-only bookkeeping and never convert FAILED to another successful state.

After exact _distutils_hack source execution finishes and its ordinary module
identity/metadata are verified, but before control returns to override's importer,
the owner holds and verifies originals `insert_shim`, `_remove_shim`, the exact
DISTUTILS_FINDER, its class, `spec_for_distutils` and `is_cpython`. Require exact
FunctionType/code object/globals bindings from the captured compiled source code
tree, not names/filenames alone. The source loader retains that compilation's
code tree, so plain function/class code constants can be matched by identity.
Check the relevant raw module/class dictionaries and instance identity before
every call; reject changed code/globals/finder fields even with copied hashes.

Retain both replacement functions before assigning them to the two exact module
attributes. Retain originals independently and track each successful assignment;
partial installation still has an unwind obligation. No other setuptools function
is patched. `_distutils_hack.override` then executes its actual source. Its call to
insert_shim is accepted exactly once, from the active override execution on the
original thread/guard, stateHOOKED, unchanged topology, both installed wrappers
identical. It opens SHIM_ACTIVE without invoking original insert_shim. Reentry,
second entry, add_shim outside this interval, alternate requester or foreign
thread poisons. No helper ever places a foreign finder on sys.meta_path.

The custodian handles only fullname=`distutils`, path=None, target=None in this
interval. It directly invokes the held original spec_for_distutils with the exact
finder self; it never delegates generic find_spec. Its nested import of
setuptools._distutils uses the existing importer and exact inventory. Retain
the producer's returned object before validation. None, wrong spec, changed
loader/class/code/closure, wrong origin or unexpected source package fails closed.
Verify the nested DistutilsLoader code constants against the held producer code,
and the create method's closure against the actual already-held source module.
Retain producer, custom loader, actual spec and source module before adoption.
Use HOST_OWNER external_import_owner for these new helper objects; reuse the
source module's existing adoption, never manufacture a second module identity.

Replace only returned spec.loader with the session's retained exact alias wrapper,
holding the original custom loader and original loader binding. The real custom
loader's create_module and exec_module still run synchronously; they are not
reimplemented. The wrapper provides the checks/retention transitions in section5.
This Python-only alias operation has its own observation below; do not mislabel
it as an ExtensionFileLoader/native create or an ordinary new source-file execution.
Nested ordinary file imports retain their existing import-call ancestry.

At `_remove_shim` entry, the outer importlib import has returned: validate the
actual alias publication, emit ALIAS_PUBLISHED, then SHIM_CLOSED. Do not call
original _remove_shim, because no shim was installed. If unwinding an exception,
record only failed cleanup, retain visible aliases/late owners, close the internal
interval and preserve failure; do not emit successful publication/closure.
The source continues its actual distutils.core import and assertions. After
override's source execution/metadata check completes, restore both originals
once, verify both actual bindings and singleton topology, then RESTORED.
Installation or override failure runs the same restoration owner from its finally
path; no second restorer is added to compiler unwind. The outer compiler only
requires this owner finalized/restored or irreversibly failed before proceeding.

COMPLETE occurs only after the actual cpp_extension import returns, setuptools
and build_ext are completed admitted records, all expected imports/absence facts,
alias state, resource/version result, original hooks, paths and environment audit
pass. It precedes the first load_inline call. Guards7/8 audit the retained COMPLETE
owner, both restored hooks and alias/path/resource bindings; they cannot renew it.

## 5. Same-object alias and metadata matrix

Only root setuptools._distutils -> distutils is a special alias. Capture source
object M, its source wrapper Ls, source spec Ss and path D at ordinary completion.
Source record/spec/loader evidence remains immutable history. Before alias create:
M.__name__ and M.__package__ are setuptools._distutils; M.__spec__ is Ss;
M.__loader__ is Ls; M.__file__ is the pinned __init__.py; M.__path__ equals [D].

Actual original custom create_module(Sa) changes only M.__name__ to distutils and
returns M. Retain the result before checking it is M. At this returned boundary,
source spec/loader/package/path still have the old values. State ALIAS_CREATED
permits this exact intermediate matrix only within this active import. Importlib
module_from_spec subsequently assigns Sa to __spec__; its existing __loader__,
__package__, __file__, __path__ survive. At alias exec entry/return require:

- M is the original source object under source name, and sys.modules['distutils']
  is M; no replacement/copy. Alias wrapper, actual custom loader and Sa are exact.
- M.__name__=`distutils`, M.__spec__ is Sa, M.__loader__ is Ls,
  M.__package__=`setuptools._distutils`, file/path unchanged.
- Sa.name=`distutils`, Sa.origin equals source file, Sa.loader is alias wrapper,
  Sa.submodule_search_locations is None, Sa.parent is the empty string; original
  source spec Ss still has original source name/loader/package locations.

The original custom exec_module is a pinned no-op but is actually invoked once.
After it returns, ALIAS_EXECUTED is recorded. Publication is checked again when
remove_shim is reached (after importlib return), then on every ordinary audit.
No later mutation of the matrix is permitted. Missing/deleted alias, replacement
source entry, changed parent path/spec/loader or changed hook binding fails closed.

The source package's normal check_module delegates only its metadata transition
to this exact owner; byte/loader/source identity checks remain shared. The alias
record kind is private `python_package_alias`, not existing native creator-self
or module replacement. Its parent lookup returns D only after successful alias
exec/publication while the owner permits the state. Each declared distutils child
then resolves through standard PathFinder plus ordinary file-wrapper verification.
Source and alias roots share M; differently named descendants are not automatically
the same objects. Forbid any unlisted counterpart, including source's asserted
absent setuptools._distutils.log. Other publications retain existing rules.

## 6. Finite resource and named metadata

Resource: only jaraco.text / `Lorem ipsum.txt`,1335bytes, expected hash from the
inventory and direct observed bytes. `_ExternalSourceLoader.get_resource_reader`
returns a held reader only for the exact admitted package and transaction binding;
all unrelated existing loaders retain their former behavior. Use the stdlib
ResourceReader compatibility interface: open_resource returns a read-only BytesIO
of the held verified bytes; resource_path rejects filesystem-path export;
is_resource recognizes exactly this one name; contents exposes only this one name.
Reject absolute paths, separators, traversal, symlinks, missing/changed bytes and
all unlisted resource names. Initial capture and use audit reuse existing contained
file/hash checks. Reads require the actual jaraco.text execution/thread/guard and
active initialization; later copied/rebound readers do not gain authority. Retain
reader, bytes and returned handle before checks/adoption/return can fail. Default
stdlib files(...).joinpath(...).read_text(encoding='utf-8') must run unchanged and
consume these verified bytes; no replacement of jaraco.text or its string result.

Metadata choice: faithfully return version83.0.0 from pinned PKG-INFO; do not
accept silent `0.dev0+unknown` as completed initialization. The same custodian
adds find_distributions. Outside this transaction or for other names, return the
existing empty discovery result (no filesystem search). For the sole accepted
request require an actual target-stdlib DistributionFinder.Context, exactly
vars(context)=={'name':'setuptools'}, unchanged expected path, actual executing
setuptools.version module and active guard/thread. No user-supplied path/context
fields, unnamed enumeration or entry-point loading is authorized.

Return a held private stdlib Distribution subclass backed by verified metadata
bytes, not a PathDistribution exposing the directory. read_text('METADATA')
returns None only after proving that exact metadata path absent; read_text('PKG-INFO')
returns the held UTF-8 text. Any other filename/locate_file attempt poisons and
raises. No entry_points.txt, SOURCES.txt or other data becomes readable. Stdlib
Distribution.metadata/version parsing runs unchanged. Record the actual parsed
Name/Version and check setuptools.version.__version__ and setuptools.__version__
are83.0.0 at COMPLETE. Read denial remains sticky even if dependency catches it.
Do not add general metadata enumeration for unproven Distribution/setup methods.

Both resources use the shared environment file table and actual token/adoption
mechanism. No new filesystem root, callback, interpreter plugin or module import
permission is granted by possession of a reader or metadata descriptor.

## 7. Observations, failure and acceptance joins

Version CUDA compilation record/selection to v2 for the new grammar; retain v1
as historical descriptive format without granting new live authority. Original
runtime-custody v2 input prefix, lifecycle cut and source trace remain byte-for-byte
unchanged. Environment v6 is bound before that cut but has no active transaction
events in it. New `setuptools_initialization` journal kind is legal only in the
compilation suffix; its details are `action`, `initialization_token`, plus exact
action fields below (no unknown fields):

| action | additional fields |
|---|---|
| begun | vendor_root, original_paths, active_paths |
| hooked | hack_module_token, finder_token, insert_original_token, remove_original_token, insert_wrapper_token, remove_wrapper_token, producer_token |
| shim_entered | override_module_token |
| alias_spec | source_module_token, source_spec_token, source_loader_token, alias_spec_token, delegate_token, wrapper_token |
| alias_created / alias_executed / alias_published | source_module_token, alias_spec_token, delegate_token, wrapper_token |
| shim_left / hooks_restored | no extra fields |
| metadata_read | distribution_token, bytes_token, file_root, file_path, sha256, version |
| resource_read | package_token, reader_token, bytes_token, handle_token, file_root, file_path, sha256 |
| completed | setuptools_module_token, build_ext_module_token, cpp_extension_module_token |
| failed | exception_token |

Journal event position is the sequence; no parallel token or sequence registry.
All referenced actual objects are strongly retained before journal serialization
or adoption. Existing compile_owner_retained events must precede first use. Reuse
HOST_OWNER external_import_owner for successful helper objects, exact guard6
anchors; the source object carries its existing file adoption. Failed partial
objects may be root-only, never satisfy successful required roles. File/module
events remain ordinary; alias facts do not increment/fabricate ordinary file-load
or legacy creator-publication counts. Transaction failure emits existing source
failure and failed transaction event when possible; diagnostic issuance failure
must not overwrite the earliest error or clear the actual failed state.

Replay validates exact action fields/order and same tokens at each alias step,
one transaction, one shim interval, both restored originals, exact deployment
file references/digests and actual lifecycle adoptions. COMPLETE requires at least
one matching metadata_read and resource_read with the same held owners (each
repeat must be a real checked read under the same execution), all three completed
module identities joined to ordinary external/file events, and closed import
stacks. No compile success/hook installation/load_inline event may precede it.
Coherent copied token labels cannot replace actual source/lifecycle/retention
joins. Old input facts cannot satisfy newly required shim/reader/helper roles.

On any failure, the session, external admission, initialization owner and compiler
remain failed through their existing chain. One owned restoration pass attempts
each actually installed hook's restoration once, verifies actual resulting
identity, audits topology on exit, and retains every error. Wrong-current-binding
is failure even if the subsequent restoration succeeds. An uncertain restoration
never yields COMPLETE. Already executing parents may catch errors and publish
late objects; existing boundary retention plus this owner's exact alias retention
holds them before reporting. No unimport, owner deletion, metadata rollback,
new setup guard, deadline renewal or successful state after failure. Supervisor
and runtime quarantine/terminal handling remain the previous design's authority.

## 8. Verification and structural acceptance

Implement only after independent approval of this amendment. Then require:

1. Production-boundary CPU integration under isolated target Python3.12, with
   actual pinned setuptools initialization and its finite resource/version path
   after a dedicated reviewed CPU preflight for that exact source/environment/
   runner. No Torch/backend import or compiler is needed for this dependency-only
   proof. Before that preflight, only stdlib/AST stand-ins and source reads run.
   Do not use a fake setuptools package as full integration proof.
2. Positive same-object alias/source-child resolution; real resource expression
   and version83.0.0; stable singleton finder and frozen P+vendor; restored hooks;
   unchanged ordinary file/publication behavior and input-cut parsing.
3. Counterexamples for preloads, vendor shadow/extra paths, foreign shim, swapped
   function code/closure, wrong spec/module/loader, early/repeated/foreign-thread
   calls, missing alias, unlisted child, traversal/resource change, metadata path
   and enumeration misuse, caught failure, partial installation/create/exec,
   restoration failure and late publication. Exercise shared production paths,
   not another hand-written importer or only synthetic successful records.
4. New record/parser joins: missing restore/resource/version, duplicate action,
   wrong guard, source/alias swap, reused input token, coherent token permutation,
   incomplete failed prefixes and attempted success after poison.
5. Fresh exact code/source/catalog/runner/observer/ABI preflight still precedes
   all new compiler/backend/native execution. Then existing actual three cold
   builds, failure/process/target1-ambient0 matrices and whole-checkpoint review.
   This amendment does not substitute CPU dependency proof for any of them.

Code review must check one owner for dependency-specific behavior, shared ordinary
loading/file/retention logic, and no independently drifting alias or failure stacks.
The later whole-C4 systematic refactor remains required after functional and real
evidence, before final acceptance. If a future approved dependency/artifact route
removes these behaviors, remove this transaction; do not retain both mechanisms.
