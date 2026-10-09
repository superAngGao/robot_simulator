"""Full initialization journal mutations through production continuation dispatch.

All rows are descriptive CPU fixtures. Guardian adoption records are real, while
the outer guard/native/input cut is a stand-in. No dependency or native execution,
full CudaCompilationRecord acceptance, or live initialization is claimed here.
"""

import copy
import hashlib
import itertools
from types import SimpleNamespace

import pytest

from optical_worker.compilation_contract import setuptools_initialization_descriptor, setuptools_role_anchor
from optical_worker.cuda_compilation_evidence import _CompilationReplay, validate_cuda_compilation_selection
from optical_worker.lifecycle import HOST_OWNER, NativeWorkerResourceGuardian
from optical_worker.protocol import canonical_json_bytes
from optical_worker.runtime_custody import _DETAIL_FIELDS
from optical_worker.wait_protocol import GuardProtocolError


class Journal:
    def __init__(self):
        self.guard = {"sequence": 6, "role": "setup"}
        self.events, self.trace, self.objects, self.exported = [], [], {}, {}
        self.guardian = NativeWorkerResourceGuardian(session_identity="setuptools_replay_fixture")
        self.owners = []
        self.descriptor = setuptools_initialization_descriptor()
        self.roots = {"site": "/env/lib/python3.12/site-packages",
                      "setuptools_vendor": "/env/lib/python3.12/site-packages/setuptools/_vendor"}
        self.files, self.rules = {}, {}
        for key in ("resource", "metadata"):
            row = self.descriptor[key]
            self.files[row["root"], row["path"]] = dict(root=row["root"], path=row["path"], sha256="a" * 64)
        self.add_adoption(1, "cuda_compile_custody", "cuda_compile_custody")
        self.exported[1] = dict(self.objects[1], origin="continuation", retention_event_index=None)
        self.replay = self.reader()
        self.emit("compile_root_registered", root_token=1)
        self.initialization = self.owner("setuptools_initialization", "setuptools_initialization")

    def reader(self):
        r = object.__new__(_CompilationReplay)
        r.root = {"journal": self.events}
        r.environment = SimpleNamespace(setuptools_initialization=self.descriptor)
        r.inputs = {"interpreter_alias": {"interpreter_identity": [None] * 6 + ["posix"]}}
        r.roots, r.files, r.rules = self.roots, self.files, self.rules
        r.objects, r.exported, r.trace = self.objects, self.exported, self.trace
        r.adoptions = {e["details"]["sequence"]: e for e in self.trace}
        r.guards = [(self.guard, True) for _ in self.trace]
        r.first, r.retained = {}, []
        r.index = r.last_lifecycle = r.lci = -1
        r.cut, r.external_count, r.completed = 0, 0, 0
        r.root_token, r.guard = None, None
        r.failed = r.setuptools_failed = False
        r.bound_environment = r.setuptools_required = True
        r.phases = list(range(6))
        r.steps = [[("build_entered", 0)], [], []]  # Outer compile leaf is descriptive.
        r.compile_native = [(0, "cuda_compile", "before")]
        r.setuptools_token = r.setuptools_alias = None
        r.setuptools_actions, r.setuptools_helpers = [], {}
        r.setuptools_reads = {"metadata_read": 0, "resource_read": 0}
        r.calls, r.call_keys, r.delegate_stack = [], {}, []
        r.file_states, r.file_owners, r.file_resolutions = {}, {}, {}
        r.returned_files, r.namespaces, r.publications = {}, {}, {}
        r.publication_rules, r.disabled, r.disabled_rules = {}, {}, {}
        return r

    def add_adoption(self, token, role, operation):
        value = object()
        self.owners.append(value)
        adopted = self.guardian.adopt(value, role=role, device_scope=HOST_OWNER, operation_id=operation)
        event = {"index": len(self.trace), "guard": self.guard, "operation": "owner_adopted", "details": adopted.to_payload()}
        self.trace.append(event)
        self.objects.setdefault(token, {"token": token, "first_journal_index": len(self.events)})["guardian_sequence"] = adopted.sequence
        if hasattr(self, "replay"):
            self.replay.adoptions[adopted.sequence] = event
            self.replay.guards.append((self.guard, True))
        return adopted.sequence

    def owner(self, role="external_import_owner", operation="external_import_owner"):
        token = len(self.objects) + 1
        self.objects[token] = {"token": token, "first_journal_index": len(self.events), "guardian_sequence": None}
        self.exported[token] = dict(self.objects[token], origin="continuation", retention_event_index=len(self.events))
        self.emit("compile_owner_retained", root_token=1, owner_token=token, slot=len(self.replay.retained))
        sequence = self.add_adoption(token, role, operation)
        self.exported[token]["guardian_sequence"] = sequence
        return token

    def helper(self, action, field, index=None):
        return self.owner(*setuptools_role_anchor(action, field, index))

    def emit(self, kind, **details):
        event = {"index": len(self.events), "kind": kind, "guard": self.guard,
                 "lifecycle_event_index": len(self.trace) - 1, "details": details}
        self.events.append(event)
        self.replay.apply_continuation(event)

    def action(self, action, **details):
        self.emit("setuptools_initialization", action=action, initialization_token=self.initialization, **details)

    def external(self, name, operation, module):
        spec, loader, origin, kind = self.resolutions[name]
        self.emit("external", event={"index": self.replay.external_count, "operation": operation,
                                     "name": name, "kind": "file_module", "phase": "runtime"},
                  provenance={"origin": origin, "delegate_kind": kind, "spec_token": spec,
                              "loader_token": loader, "module_token": module})

    def file_start(self, name):
        path = name.replace(".", "/") + ".py"
        self.rules[name] = {"root": "site", "path": path, "phase": "runtime"}
        self.files["site", path] = {"kind": "python"}
        spec, loader = self.owner(), self.owner()
        self.resolutions[name] = spec, loader, self.roots["site"] + "/" + path, "SourceFileLoader"
        self.external(name, "file_resolved", None)
        self.external(name, "create_started", None)
        frame = {"sequence": len(self.replay.calls), "observer": name, "operation": "create", "state": "started",
                 "phase": "runtime", "spec_token": spec, "loader_token": loader, "module_token": None,
                 "parent_sequence": self.replay.delegate_stack[-1] if self.replay.delegate_stack else None}
        self.emit("file_call", **frame)
        module = self.owner()
        self.emit("returned_owner", producer="file_create", object_token=module,
                  call_sequence=frame["sequence"], leaf_event_index=None)
        self.emit("file_call", **dict(frame, state="returned", module_token=module))
        self.external(name, "created", module)
        self.emit("file_call", **dict(frame, state="completed", module_token=module))
        self.external(name, "exec_started", module)
        self.emit("file_call", **dict(frame, sequence=len(self.replay.calls), operation="exec", module_token=module))
        return module

    def file_end(self, name):
        call = self.replay.calls[self.replay.call_keys[name, "exec"]]
        frame = {key: call[key] for key in _DETAIL_FIELDS["file_call"]}
        self.emit("file_call", **dict(frame, state="returned"))
        self.emit("file_call", **dict(frame, state="completed"))
        self.external(name, "completed", frame["module_token"])

    def build(self, *, resource_reads=2, metadata_reads=2):
        self.resolutions = {}
        self.action("begun", vendor_root="setuptools_vendor", original_paths=["/env/lib/python3.12"],
                    active_paths=["/env/lib/python3.12", self.roots["setuptools_vendor"]])
        cpp = self.file_start("torch.utils.cpp_extension")
        setuptools = self.file_start("setuptools")
        hack = self.file_start("_distutils_hack")
        self.file_end("_distutils_hack")
        fields = ("finder_token", "producer_token", "insert_original_token", "remove_original_token",
                  "insert_wrapper_token", "remove_wrapper_token")
        self.action("hooked", hack_module_token=hack, **{field: self.helper("hooked", field) for field in fields})
        override = self.file_start("_distutils_hack.override")
        self.action("shim_entered", override_module_token=override)
        source = self.file_start("setuptools._distutils")
        self.file_end("setuptools._distutils")
        alias = {field: self.helper("alias_spec", field) for field in ("alias_spec_token", "delegate_token", "wrapper_token")}
        alias["source_module_token"] = source
        spec, loader = self.resolutions["setuptools._distutils"][:2]
        self.action("alias_spec", **alias, source_spec_token=spec, source_loader_token=loader)
        for action in ("alias_created", "alias_executed", "alias_published"):
            self.action(action, **alias)
        self.action("shim_left")
        self.file_end("_distutils_hack.override")
        self.action("hooks_restored")
        package = self.file_start("jaraco.text")
        reader, data = (self.helper("resource_read", field, 0) for field in ("reader_token", "bytes_token"))
        for index in range(resource_reads):
            self.action("resource_read", read_index=index, package_token=package, reader_token=reader, bytes_token=data,
                        handle_token=self.helper("resource_read", "handle_token", index),
                        file_root="setuptools_vendor", file_path=self.descriptor["resource"]["path"], sha256="a" * 64)
        self.file_end("jaraco.text")
        self.file_start("setuptools.version")
        distribution, data = (self.helper("metadata_read", field, 0) for field in ("distribution_token", "bytes_token"))
        for index in range(metadata_reads):
            self.action("metadata_read", read_index=index, distribution_token=distribution, bytes_token=data,
                        file_root="site", file_path=self.descriptor["metadata"]["path"], sha256="a" * 64, version="83.0.0")
        self.file_end("setuptools.version")
        self.file_end("setuptools")
        build_ext = self.file_start("setuptools.command.build_ext")
        self.file_end("setuptools.command.build_ext")
        self.file_end("torch.utils.cpp_extension")
        self.action("completed", setuptools_module_token=setuptools, build_ext_module_token=build_ext, cpp_extension_module_token=cpp)
        return self


@pytest.fixture
def journal():
    return Journal().build()


def replay(journal, events=None):
    reader = journal.reader()
    events = journal.events if events is None else events
    reader.root = {"journal": events}
    for event in events:
        reader.apply_continuation(event)
    return reader


def test_complete_initialization_grammar_and_ordinary_file_joins(journal):
    reader = replay(journal)
    assert reader.setuptools_complete and not reader.failed
    assert reader.setuptools_reads == {"resource_read": 2, "metadata_read": 2}
    assert len(reader.setuptools_helpers) == 15 and reader.completed == 0
    assert not reader.delegate_stack


def test_all_coherent_helper_permutations_fail_against_fixed_lifecycle(journal):
    tokens = list(journal.replay.setuptools_helpers.values())
    for left, right in itertools.combinations(tokens, 2):
        events = copy.deepcopy(journal.events)
        for event in events:
            if event["kind"] == "setuptools_initialization":
                for field, token in event["details"].items():
                    if field.endswith("_token") and token in (left, right):
                        event["details"][field] = right if token == left else left
        with pytest.raises(ValueError, match="adoption operation mismatch|prior continuation retention"):
            replay(journal, events)


@pytest.mark.parametrize("action,field,value", [
    ("begun", "active_paths", ["/foreign"]), ("metadata_read", "version", "0.dev0+unknown"),
    ("metadata_read", "file_path", "entry_points.txt"), ("resource_read", "read_index", 1),
    ("alias_spec", "source_loader_token", 1), ("completed", "setuptools_module_token", 1),
    ("hooks_restored", "action", "completed"),
    ("alias_created", "action", "alias_executed"),
])
def test_action_and_source_binding_mutations_reject(journal, action, field, value):
    events = copy.deepcopy(journal.events)
    target = next(e for e in events if e["kind"] == "setuptools_initialization" and e["details"]["action"] == action)
    target["details"][field] = value
    with pytest.raises((ValueError, GuardProtocolError)):
        replay(journal, events)


def test_same_guard_unrelated_source_cannot_replace_completed_module(journal):
    events = copy.deepcopy(journal.events)
    target = events[-1]["details"]
    target["setuptools_module_token"] = journal.replay.file_owners["jaraco.text"]
    with pytest.raises(ValueError, match="ordinary source identity"):
        replay(journal, events)


def test_current_grammar_is_not_available_to_historical_v1(journal):
    reader = journal.reader()
    reader.setuptools_required = False
    with pytest.raises(ValueError, match="outside compilation v2"):
        for event in journal.events:
            reader.apply_continuation(event)


@pytest.mark.parametrize("kind", ["resource_reads", "metadata_reads"])
def test_complete_rejects_missing_actual_read_observations(kind):
    with pytest.raises(ValueError, match="completion lacks reads"):
        Journal().build(**{kind: 0})


def test_helper_retention_must_precede_its_adoption_not_only_its_action(journal):
    events = copy.deepcopy(journal.events)
    token = journal.replay.setuptools_helpers["setuptools_initialization.finder"]
    row = journal.exported[token]
    adoption = journal.replay.adoptions[row["guardian_sequence"]]
    events[row["retention_event_index"]]["lifecycle_event_index"] = adoption["index"]
    with pytest.raises(ValueError, match="retention order"):
        replay(journal, events)


def test_ordinary_source_cannot_be_reused_from_input_cut(journal):
    token = journal.replay.file_owners["setuptools._distutils"]
    journal.exported[token]["origin"] = "input"
    with pytest.raises(ValueError, match="retention"):
        replay(journal)


def test_metadata_read_must_be_inside_its_own_active_exec_call(journal):
    reader = journal.reader()
    for event in journal.events:
        if event["kind"] == "setuptools_initialization" and event["details"]["action"] == "metadata_read":
            # Simulate a still-existing completed record, while another parent
            # is current. Merely naming the right module cannot grant a read.
            reader.delegate_stack.pop()
            with pytest.raises(ValueError, match="exact executing source"):
                reader.apply_continuation(event)
            return
        reader.apply_continuation(event)
    raise AssertionError("fixture did not exercise metadata")


def test_partial_unadopted_initialization_can_fail_but_cannot_issue_success():
    journal = Journal()
    journal.objects[journal.initialization]["guardian_sequence"] = None
    journal.exported[journal.initialization]["guardian_sequence"] = None
    error = journal.owner()
    journal.action("failed", exception_token=error)
    assert journal.replay.failed and not journal.replay.setuptools_complete
    with pytest.raises(ValueError, match="outside original active guard"):
        journal.action("begun", vendor_root="setuptools_vendor", original_paths=["/env/lib/python3.12"],
                       active_paths=["/env/lib/python3.12", journal.roots["setuptools_vendor"]])


@pytest.mark.parametrize("version", [1, 2])
def test_selection_cannot_relabel_a_record_as_another_grammar_version(version):
    # Selector proof only; this minimal descriptor is not a compilation record.
    record = canonical_json_bytes({"schema_version": f"q54.p12.4.cuda-compilation.v{version}",
                                   "observation_cut": {"lifecycle_event_count": 5}})
    selection = {"schema_version": f"q54.p12.4.cuda-compilation-selection.v{version}",
                 "selection": "current", "record_sha256": hashlib.sha256(record).hexdigest(),
                 "selected_lifecycle_event_count": 5, "observed_lifecycle_event_count": 5,
                 "failure_diagnostic_present": False}
    validate_cuda_compilation_selection(selection, record, observed_lifecycle_event_count=5)
    selection["schema_version"] = f"q54.p12.4.cuda-compilation-selection.v{3 - version}"
    with pytest.raises(ValueError, match="record mismatch"):
        validate_cuda_compilation_selection(selection, record, observed_lifecycle_event_count=5)
