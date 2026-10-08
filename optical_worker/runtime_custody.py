"""CPU-only runtime observations. Descriptors never grant live authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from .guarded_evidence import GuardedLifecycleEvidence, _Replay as _LifecycleReplay, validate_custody_artifact
from .lifecycle import HOST_OWNER
from .native_manifest import CANONICAL_DTYPE_METADATA, NativeInputManifest
from .protocol import canonical_json_bytes, canonical_sha256, decode_canonical_message
from .runtime_environment import (
    INTERPRETER_ALIAS_OBSERVATIONS_VERSION,
    RuntimeEnvironmentContract,
    validate_interpreter_alias_observations,
    validate_module_replacement_observations,
    validate_native_global_observations,
)
from .wait_protocol import GuardConfiguration, GuardProtocolError, exact_fields

RUNTIME_CUSTODY_VERSION = "q54.p12.4.runtime-custody.v1"
RUNTIME_CUSTODY_INPUTS_VERSION = "q54.p12.4.runtime-custody-inputs.v1"
RUNTIME_CUSTODY_SELECTION_VERSION = "q54.p12.4.runtime-custody-selection.v1"
PHASES = (
    "setup_definitions", "external_environment", "canonical_materialization",
    "runtime_initialization", "native_upload", "input_validation",
)

_DETAIL_FIELDS = {
    "source_bound": {"session_token", "bootstrap_token", "controller_token"},
    "environment_bound": {"environment_sha256", "environment_token", "admission_token"},
    "external": {"event", "provenance"},
    "file_call": {"sequence", "parent_sequence", "observer", "operation", "phase", "state",
                  "module_token", "spec_token", "loader_token"},
    "native_global": {"event_index", "token_bindings", "adoption_event_index"},
    "module_replacement": {"event_index", "token_bindings"},
    "interpreter_alias": {"event_index", "token_bindings", "publisher_token", "spec_token", "loader_token"},
    "runtime_binding": {"binding_token", "controller_token", "module_token", "runtime_token",
                        "target_token", "stream_token", "source_token", "stream_identity"},
    "binding_consumed": {"binding_token"},
    "prepared_storage": {"prepared_token", "source_token", "ready_event_token", "manifest", "arrays",
                         "storage_sha256"},
    "input_validated": {"prepared_token", "manifest_sha256", "content_sha256", "topology_sha256",
                        "validated_arrays"},
    "phase_completed": {"phase", "disarmed_event_index", "materialization_case_id"},
    "failure": {"source", "exception_token"},
    "returned_owner": {"producer", "call_sequence", "leaf_event_index", "object_token"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "invalid runtime custody integer")
    return value


def _text(value):
    _require(type(value) is str and bool(value), "invalid runtime custody text")
    return value


def _digest(value, length=64):
    _require(type(value) is str and len(value) == length
             and all(c in "0123456789abcdef" for c in value), "invalid runtime custody digest")
    return value


def _source(value):
    p = exact_fields(value, {"revision", "catalog_sha256", "snapshot_sha256"})
    _digest(p["revision"], 40)
    _digest(p["catalog_sha256"])
    _digest(p["snapshot_sha256"])
    return p


def validate_runtime_custody_inputs(value, environment):
    """Compare independently captured launch intent without filesystem access."""
    _require(type(environment) is RuntimeEnvironmentContract, "runtime custody environment type")
    p = exact_fields(value, {"schema_version", "source", "lane_id", "configuration", "case_id",
                             "interpreter_alias"})
    _require(p["schema_version"] == RUNTIME_CUSTODY_INPUTS_VERSION, "runtime custody inputs version")
    _source(p["source"])
    _require(type(p["lane_id"]) is str and p["lane_id"] in {"warp", "cuda"}, "runtime custody lane")
    GuardConfiguration.from_payload(p["configuration"])
    _text(p["case_id"])
    alias = exact_fields(p["interpreter_alias"], {"interpreter_identity", "resolved_executable",
                         "executable_sha256", "source_path", "source_sha256"})
    # The existing parser is the one lexical identity/source topology policy.
    validate_interpreter_alias_observations({
        "schema_version": INTERPRETER_ALIAS_OBSERVATIONS_VERSION,
        "interpreter_identity": alias["interpreter_identity"],
        "source_path": alias["source_path"], "source_sha256": alias["source_sha256"],
        "resolution": None, "events": [],
    })
    _text(alias["resolved_executable"])
    _digest(alias["executable_sha256"])
    interpreter = environment.to_payload()["interpreter"]
    _require(alias["resolved_executable"] == interpreter["path"]
             and alias["executable_sha256"] == interpreter["sha256"]
             and alias["interpreter_identity"][1][:3] == interpreter["version"],
             "launch interpreter differs from runtime environment")
    canonical_json_bytes(p)


def _validate_storage_signature(value, array, lane, target):
    p = exact_fields(value, {"device_ordinal", "base_pointer", "storage_identity", "storage_nbytes",
                            "active_start", "active_count", "dtype", "shape", "strides"})
    for key in ("device_ordinal", "base_pointer", "storage_identity", "storage_nbytes",
                "active_start", "active_count"):
        _integer(p[key])
    _require(p["device_ordinal"] == target, "storage device differs from guarded target")
    _require(type(p["shape"]) is list and type(p["strides"]) is list,
             "storage shape/strides must be arrays")
    for dimension in p["shape"] + p["strides"]:
        _integer(dimension)
    _require(p["shape"] == list(array.shape) and len(p["strides"]) == len(array.shape)
             and p["active_count"] == array.nbytes, "storage extent differs from canonical array")
    _require(p["active_start"] + p["active_count"] <= p["storage_nbytes"], "storage range overflow")
    _require(p["storage_nbytes"] == 0 or (p["base_pointer"] > 0 and p["storage_identity"] > 0),
             "nonempty storage needs physical pointer/identity")
    metadata = CANONICAL_DTYPE_METADATA[array.dtype]
    _require(p["dtype"] == metadata[1 if lane == "cuda" else 2], "storage dtype differs from manifest")
    if array.nbytes:
        stride = 1 if lane == "cuda" else metadata[0]
        for dimension, actual in zip(reversed(array.shape), reversed(p["strides"])):
            if dimension > 1:
                _require(actual == stride, "storage strides are not canonical contiguous layout")
            stride *= dimension
    # Zero-size signatures remain the actual registered backend observation;
    # their strides/pointer are not replaced with invented physical storage.


def validate_runtime_custody_selection(value, record_bytes, *, observed_lifecycle_event_count=None):
    """Check a durable pair; never infer a record from a newer sidecar.

    This checks persistence consistency only. Call the aggregate/attachment
    parser separately with independently supplied launch inputs and environment.
    """
    p = exact_fields(value, {"schema_version", "selection", "runtime_record_sha256",
                            "selected_lifecycle_event_count", "observed_lifecycle_event_count",
                            "failure_diagnostic_present"})
    _require(p["schema_version"] == RUNTIME_CUSTODY_SELECTION_VERSION, "runtime selection version")
    _require(type(p["selection"]) is str
             and p["selection"] in {"current", "prior_validated_cut", "unavailable"}, "runtime selection kind")
    observed = _integer(p["observed_lifecycle_event_count"])
    _require(type(p["failure_diagnostic_present"]) is bool, "runtime selection diagnostic flag")
    if observed_lifecycle_event_count is not None:
        _integer(observed_lifecycle_event_count)
        _require(observed == observed_lifecycle_event_count, "runtime selection observed count mismatch")
    if p["selection"] == "unavailable":
        _require(record_bytes is None and p["runtime_record_sha256"] is None
                 and p["selected_lifecycle_event_count"] is None, "unavailable selection carries a record")
        return
    _require(type(record_bytes) is bytes, "runtime selection lacks durable record bytes")
    _digest(p["runtime_record_sha256"])
    _require(hashlib.sha256(record_bytes).hexdigest() == p["runtime_record_sha256"],
             "runtime durable record digest mismatch")
    record = decode_canonical_message(record_bytes, max_bytes=len(record_bytes))
    _require(record.get("schema_version") == RUNTIME_CUSTODY_VERSION, "runtime durable record version")
    lifecycle = GuardedLifecycleEvidence.from_payload(record["lifecycle"]).to_payload()
    count = _integer(p["selected_lifecycle_event_count"], 1)
    _require(count == len(lifecycle["transition_trace"])
             and record["observation_cut"]["lifecycle_event_count"] == count and count <= observed,
             "runtime durable record lifecycle count mismatch")
    if p["selection"] == "current":
        _require(count == observed, "current selection is an older lifecycle cut")


class _RuntimeReplay:
    """Cross-joins descriptive observations; lifecycle replay remains external."""

    def __init__(self, root, environment, inputs):
        self.root, self.environment, self.inputs = root, environment, inputs
        self.config = GuardConfiguration.from_payload(root["configuration"])
        self.lane = root["lane_id"]
        self.lifecycle = GuardedLifecycleEvidence.from_payload(root["lifecycle"]).to_payload()
        _require(self.lifecycle["configuration"] == root["configuration"]
                 and self.lifecycle["lane_id"] == self.lane, "runtime lifecycle identity mismatch")
        self.trace = self.lifecycle["transition_trace"]
        self.normal_guards = [e["guard"] for e in self.trace if e["operation"] == "normal_arm_started"]
        self.leaf_operations = {}
        operation = None
        for event in self.trace:
            if event["operation"] == "setup_operation_started":
                operation = event["details"]["operation_id"]
            elif event["operation"] == "low_level_leaf_entered":
                self.leaf_operations[event["index"]] = operation
        self.adoptions = {e["details"]["sequence"]: e for e in self.trace if e["operation"] == "owner_adopted"}
        replay = _LifecycleReplay(self.config, self.lane)
        self.guards = []
        for event in self.trace:
            replay.apply(event)
            try:
                replay._active_setup()
                active = True
            except GuardProtocolError:
                active = False
            self.guards.append((None if replay.guard is None else dict(replay.guard), active))
        self.objects, self.first = {}, {}
        previous, sequences = 0, set()
        _require(type(root["objects"]) is list, "runtime objects must be an array")
        for raw in root["objects"]:
            row = exact_fields(raw, {"token", "first_journal_index", "guardian_sequence"})
            token = _integer(row["token"], 1)
            _require(token > previous, "runtime object token order")
            previous = token
            _integer(row["first_journal_index"])
            sequence = row["guardian_sequence"]
            if sequence is not None:
                _integer(sequence)
                _require(sequence in self.adoptions and sequence not in sequences,
                         "runtime guardian sequence is absent or shared by distinct objects")
                sequences.add(sequence)
            self.objects[token] = row
        self.components = exact_fields(root["components"], {"native_globals", "module_replacements", "interpreter_alias"})
        native, replacements, alias = (self.components[k] for k in ("native_globals", "module_replacements", "interpreter_alias"))
        _require((native is None) == (replacements is None), "partial initialized component pair")
        if native is not None:
            validate_native_global_observations(native, environment)
            validate_module_replacement_observations(replacements, environment, native)
        _require(alias is not None, "bound source lacks captured interpreter alias descriptor")
        validate_interpreter_alias_observations(alias)
        for key in ("interpreter_identity", "source_path", "source_sha256"):
            _require(alias[key] == inputs["interpreter_alias"][key], "runtime alias differs from launch capture")
        self.component_seen = {key: 0 for key in self.components}
        self.local_tokens = {key: {} for key in (*self.components, "disabled_optional")}
        self.source = self.binding = self.prepared = self.manifest = None
        self.bound_environment = self.consumed = self.validated = self.failed = False
        self.phases, self.phase_guards = [], []
        self.calls, self.call_keys, self.delegate_stack = [], {}, []
        self.file_states, self.file_owners, self.file_resolutions = {}, {}, {}
        self.self_absences, self.publications, self.namespaces, self.disabled = {}, {}, {}, {}
        self.returned_files, self.returned_runtime = {}, {}
        self.external_count, self.last_lifecycle, self.index, self.lci, self.guard = 0, -1, -1, -1, None
        ep = environment.to_payload()
        self.env = ep
        self.rules = {r["name"]: r for r in ep["modules"]}
        self.roots = {r["id"]: r["path"] for r in ep["roots"]}
        self.files = {(r["root"], r["path"]): r for r in ep["files"]}
        self.publication_rules = {r["name"]: r for r in ep["publications"]}
        self.namespace_rules = {r["name"]: r for r in ep["namespaces"]}
        self.disabled_rules = {r["name"]: r for r in ep["disabled_optional_modules"]}

    def token(self, value, *, nullable=False):
        if nullable and value is None:
            return None
        token = _integer(value, 1)
        _require(token in self.objects, "runtime reference lacks object row")
        self.first.setdefault(token, self.index)
        return token

    def adopted(self, token, *, roles=None, scope=None, operation=None, before=None):
        sequence = self.objects[token]["guardian_sequence"]
        _require(sequence is not None, "runtime owner has no observed direct adoption")
        event = self.adoptions[sequence]
        row = event["details"]
        _require(event["index"] <= (self.lci if before is None else before), "runtime adoption occurs after use")
        _require(roles is None or row["role"] in roles, "runtime adoption role mismatch")
        _require(scope is None or row["device_scope"] == scope, "runtime adoption scope mismatch")
        _require(operation is None or row["operation_id"] == operation, "runtime adoption operation mismatch")
        return event

    def at_phase(self, count):
        _require(len(self.phases) == count, "runtime observation occurs in wrong phase")

    def file_call(self, d):
        _require(self.bound_environment, "file call before environment binding")
        sequence = _integer(d["sequence"])
        name, operation, state = d["observer"], d["operation"], d["state"]
        _require(type(name) is str and name in self.rules and operation in {"create", "exec"}
                 and d["phase"] == self.rules[name]["phase"], "file call rule mismatch")
        module = self.token(d["module_token"], nullable=True)
        spec, loader = self.token(d["spec_token"]), self.token(d["loader_token"])
        _require(name in self.file_resolutions
                 and (spec, loader) == self.file_resolutions[name][:2], "file call resolution owner mismatch")
        if state == "started":
            _require(sequence == len(self.calls) and (name, operation) not in self.call_keys,
                     "file call sequence/uniqueness")
            parent = d["parent_sequence"]
            if parent is not None:
                _integer(parent)
            _require(parent == (self.delegate_stack[-1] if self.delegate_stack else None), "file call nesting mismatch")
            _require(self.file_states.get(name) == ("create_started" if operation == "create" else "exec_started"),
                     "file call lacks external start")
            _require(module is None if operation == "create" else module == self.file_owners.get(name),
                     "file call initial module mismatch")
            self.calls.append(dict(d, guard=self.guard, start_journal=self.index))
            self.call_keys[name, operation] = sequence
            self.delegate_stack.append(sequence)
        else:
            _require(sequence < len(self.calls), "file transition before start")
            previous = self.calls[sequence]
            for field in _DETAIL_FIELDS["file_call"] - {"state", "module_token"}:
                _require(d[field] == previous[field], "file call immutable field drift")
            allowed = {"started": {"returned", "failed"}, "returned": {"completed", "failed"},
                       "completed": {"failed"}, "failed": set()}
            _require(state in allowed[previous["state"]], "invalid file call transition")
            if state != "failed":
                _require(self.guard == previous["guard"], "file call moved between guards")
                _require(module is not None and module == (
                    self.returned_files.get(sequence) if operation == "create" else self.file_owners.get(name)
                ), "file return lacks retained exact owner")
            else:
                allowed_module = self.returned_files.get(sequence) if operation == "create" else previous["module_token"]
                _require(module == previous["module_token"] or module is not None and module == allowed_module,
                         "failed file call invented a returned module")
            if state == "returned":
                _require(self.delegate_stack and self.delegate_stack[-1] == sequence, "file return nesting mismatch")
                self.delegate_stack.pop()
            elif state == "failed" and sequence in self.delegate_stack:
                _require(self.delegate_stack[-1] == sequence, "file failure nesting mismatch")
                self.delegate_stack.pop()
            if state == "completed" and operation == "create":
                _require(self.file_states.get(name) == "created", "file completion before created observation")
            previous.update(state=state, module_token=module)
            if state == "failed":
                self.failed = True

    def _executing(self, name):
        sequence = self.call_keys.get((name, "exec"))
        _require(sequence is not None and self.calls[sequence]["state"] in {"started", "returned"}
                 and self.file_states.get(name) == "exec_started", "external outcome lacks executing requester/publisher")
        return self.calls[sequence]

    def _origin(self, rule):
        path = PureWindowsPath if self.inputs["interpreter_alias"]["interpreter_identity"][6] == "nt" else PurePosixPath
        return str(path(self.roots[rule["root"]]) / rule["path"])

    def external(self, d):
        _require(self.bound_environment, "external observation before environment")
        event, provenance = d["event"], d["provenance"]
        _require(type(event) is dict and type(provenance) is dict, "external observation payload type")
        base_fields = {"index", "operation", "name", "kind", "phase"}
        _require(base_fields <= set(event), "external event fields missing")
        _require(_integer(event["index"]) == self.external_count, "external event sequence")
        self.external_count += 1
        name, operation, kind, phase = (event[k] for k in ("name", "operation", "kind", "phase"))
        _text(name)
        _require(type(kind) is str and type(operation) is str and type(phase) is str, "external event enum types")
        _require(phase == ("materialization" if len(self.phases) <= 2 else "runtime"), "external phase differs from setup")
        if kind != "disabled_optional":
            exact_fields(event, base_fields)
        if kind == "failure":
            exact_fields(provenance, {"exception_token"})
            _require(operation == "failed", "external failure operation")
            self.token(provenance["exception_token"])
            self.failed = True
        elif kind == "file_module":
            p = exact_fields(provenance, {"origin", "delegate_kind", "spec_token", "loader_token", "module_token"})
            _require(name in self.rules and self.rules[name]["phase"] == phase, "external file name/phase")
            rule = self.rules[name]
            loader_kind = "SourceFileLoader" if self.files[rule["root"], rule["path"]]["kind"] == "python" else "ExtensionFileLoader"
            _require(p["origin"] == self._origin(rule) and p["delegate_kind"] == loader_kind, "external resolved origin/delegate mismatch")
            spec, loader = self.token(p["spec_token"]), self.token(p["loader_token"])
            _require(spec != loader, "external spec and delegated loader collide")
            self.adopted(spec, roles={"external_import_owner"}, scope=HOST_OWNER)
            self.adopted(loader, roles={"external_import_owner"}, scope=HOST_OWNER)
            module = self.token(p["module_token"], nullable=True)
            previous = self.file_states.get(name)
            if operation == "file_resolved":
                _require(name not in self.file_resolutions and module is None, "duplicate/late file resolution")
                self.file_resolutions[name] = (spec, loader, p["origin"], loader_kind)
            else:
                _require(self.file_resolutions.get(name) == (spec, loader, p["origin"], loader_kind),
                         "external resolution identity drift")
            if operation.startswith("self_alias_"):
                names = ("self_alias_precreate_absent", "self_alias_postcreate_absent", "self_alias_preexec_absent")
                checks = self.self_absences.setdefault(name, [])
                _require(len(checks) < 3 and operation == names[len(checks)] and loader_kind == "ExtensionFileLoader",
                         "self-alias absence order/delegate")
                required = ("file_resolved", "create_started", "created")[len(checks)]
                _require(previous == required and any(r["creator"] == name and not r["attributes"]
                                                     for r in self.publication_rules.values()), "unconfigured self-alias absence")
                if len(checks) == 0:
                    _require(module is None, "precreate absence has module")
                else:
                    sequence = self.call_keys.get((name, "create"))
                    _require(sequence is not None and module == self.returned_files.get(sequence), "self-alias absence module")
                checks.append(operation)
                return
            transitions = {"file_resolved": None, "create_started": "file_resolved", "created": "create_started",
                           "exec_started": "created", "completed": "exec_started"}
            _require(operation in transitions and previous == transitions[operation], "external file transition")
            if operation in {"file_resolved", "create_started"}:
                _require(module is None, "external module before actual return")
            elif operation == "created":
                sequence = self.call_keys.get((name, "create"))
                _require(sequence is not None and self.calls[sequence]["state"] == "returned"
                         and module == self.returned_files.get(sequence), "created event lacks actual file return")
                self.file_owners[name] = module
                self.adopted(module, roles={"external_import_owner"}, scope=HOST_OWNER)
            else:
                _require(module == self.file_owners.get(name), "external module substitution")
                sequence = self.call_keys.get((name, "create" if operation == "exec_started" else "exec"))
                _require(sequence is not None and self.calls[sequence]["state"] == "completed", "external event before call completion")
                if operation == "completed":
                    _require(all(n in self.publications for n, r in self.publication_rules.items() if r["publisher"] == name),
                             "completed publisher lacks required publication")
                    _require(all(state == "disabled_fallback_confirmed" for n, state in self.disabled.items()
                                 if self.disabled_rules[n]["requester"] == name), "completed requester lacks fallback")
            self.file_states[name] = operation
        elif kind == "absent_probe":
            p = exact_fields(provenance, {"requester", "requester_token"})
            requester = _text(p["requester"])
            _require(operation == "absent_probe" and any(r == {"name": name, "requester": requester, "phase": phase}
                                                        for r in self.env["absent_probes"]), "unconfigured absence probe")
            self._executing(requester)
            _require(self.token(p["requester_token"]) == self.file_owners.get(requester), "absence requester substitution")
        elif kind in {"created_module", "module_alias"}:
            p = exact_fields(provenance, {"module_token", "creator_token", "publisher_token", "relation", "attributes"})
            _require(name in self.publication_rules and name not in self.publications and operation == "creator_publication",
                     "unconfigured/duplicate creator publication")
            rule = self.publication_rules[name]
            _require(rule["kind"] == kind and rule["phase"] == phase and p["attributes"] == rule["attributes"]
                     and p["relation"] == ("creator_attribute" if rule["attributes"] else "creator_self"), "creator publication rule mismatch")
            self._executing(rule["publisher"])
            creator = self.token(p["creator_token"])
            publisher = self.token(p["publisher_token"])
            module = self.token(p["module_token"])
            known = self.file_owners | self.publications
            _require(creator == known.get(rule["creator"]) and publisher == self.file_owners.get(rule["publisher"]),
                     "creator/publisher token substitution")
            if p["relation"] == "creator_self":
                _require(module == creator == publisher and len(self.self_absences.get(rule["creator"], [])) == 3,
                         "creator-self lacks exact identity/absence checkpoints")
            self.adopted(module, roles={"external_import_owner"}, scope=HOST_OWNER)
            self.publications[name] = module
        elif kind == "namespace":
            p = exact_fields(provenance, {"module_token", "locations"})
            _require(name in self.namespace_rules and name not in self.namespaces and operation == "namespace",
                     "unconfigured/duplicate namespace")
            rule = self.namespace_rules[name]
            _require(phase == rule["phase"] and p["locations"] == [self._origin(r) for r in rule["locations"]],
                     "namespace search locations mismatch")
            module = self.token(p["module_token"])
            self.adopted(module, roles={"external_import_owner"}, scope=HOST_OWNER)
            self.namespaces[name] = module
        elif kind == "disabled_optional":
            self.disabled_optional(event, provenance)
        else:
            raise ValueError("unknown external observation kind")

    def bindings(self, namespace, rows, used):
        _require(type(rows) is list, "component token bindings must be an array")
        mapping = {}
        for value in rows:
            row = exact_fields(value, {"local_token", "object_token"})
            local = row["local_token"]
            if namespace == "disabled_optional":
                _text(local)
                prefix, separator, number = local.rpartition("-")
                _require(prefix == "external-owner" and separator and number.isascii()
                         and number.isdecimal() and str(int(number)) == number, "disabled local owner spelling")
            else:
                _integer(local, 1)
            _require(local not in mapping, "duplicate local token binding")
            token = self.token(row["object_token"])
            previous = self.local_tokens[namespace]
            _require(local not in previous or previous[local] == token, "component token rebind")
            _require(all(old == local or global_token != token for old, global_token in previous.items()),
                     "distinct local tokens collapsed to same object")
            previous[local] = mapping[local] = token
        _require(list(mapping) == sorted(used), "component bindings must cover exact used tokens in order")
        return mapping

    def component(self, kind, d):
        key = {"native_global": "native_globals", "module_replacement": "module_replacements",
               "interpreter_alias": "interpreter_alias"}[kind]
        component = self.components[key]
        index = _integer(d["event_index"])
        _require(component is not None and index == self.component_seen[key] and index < len(component["events"]),
                 "component event order/reference")
        event = component["events"][index]
        self.component_seen[key] += 1
        if kind == "interpreter_alias":
            mapping = self.bindings(key, d["token_bindings"], {event["main_token"], event["publisher_token"]})
            publisher = self.token(d["publisher_token"])
            spec, loader = self.token(d["spec_token"]), self.token(d["loader_token"])
            _require(mapping[event["main_token"]] == self.source["bootstrap_token"]
                     and mapping[event["publisher_token"]] == publisher
                     and len({publisher, spec, loader, self.source["bootstrap_token"]}) == 4,
                     "interpreter alias object/resolution substitution")
            return
        sequence = event["call_sequence"]
        _require(sequence < len(self.calls), "component observed before file call")
        call = self.calls[sequence]
        _require(call["guard"] == self.guard and call["state"] in {"started", "returned"},
                 "component event outside original file interval")
        _require(event["phase"] == call["phase"] and (event["checkpoint"] != "exec_return" or call["state"] == "returned"),
                 "component return checkpoint before actual return")
        if kind == "native_global":
            mapping = self.bindings(key, d["token_bindings"], {event["object_token"]})
            token = mapping[event["object_token"]]
            adoption = self.adopted(token, scope=HOST_OWNER, operation="native_global_observation")
            _require(_integer(d["adoption_event_index"]) == adoption["index"], "native global adoption reference mismatch")
        else:
            used = {event[k] for k in ("original", "replacement", "class", "native_root", "native_value")
                    if event[k] is not None}
            mapping = self.bindings(key, d["token_bindings"], used)
            _require(mapping[event["original"]] == self.file_owners.get(event["name"]),
                     "replacement original differs from returned file module")
            if event["native_root"] is not None:
                rule = next(r for r in self.env["module_replacements"] if r["name"] == event["name"])
                _require(mapping[event["native_root"]] == self.file_owners.get(rule["native_root"]),
                         "replacement native root differs from actual file module")
            self.adopted(mapping[event["replacement"]], roles={"external_import_owner"}, scope=HOST_OWNER,
                         operation="module_replacement_observation")

    def disabled_optional(self, event, provenance):
        fields = {"index", "operation", "name", "kind", "phase", "environment_sha256", "rule_sha256", "requester",
                  "requester_identity", "requester_source_sha256", "origin", "source_sha256", "loader_kind", "present",
                  "absent_prerequisite", "prerequisite_spec", "request_id", "fallback_attribute", "fallback_identity",
                  "create_count", "exec_count", "body_count", "native_create_count", "error_type", "error_args",
                  "error_name", "error_path"}
        exact_fields(event, fields)
        p = exact_fields(provenance, {"requester_token", "fallback_token", "token_bindings"})
        name, operation = event["name"], event["operation"]
        _require(name in self.disabled_rules, "unconfigured disabled optional")
        rule = self.disabled_rules[name]
        previous = self.disabled.get(name)
        _require((operation == "disabled_optional" and previous is None)
                 or (operation == "disabled_fallback_confirmed" and previous == "disabled_optional"),
                 "disabled optional transition")
        self._executing(rule["requester"])
        requester = self.token(p["requester_token"])
        fallback = self.token(p["fallback_token"], nullable=True)
        _require(requester == self.file_owners.get(rule["requester"])
                 and (fallback is None) == (operation == "disabled_optional"), "disabled requester/fallback identity")
        used = {event["requester_identity"]}
        if event["fallback_identity"] is not None:
            used.add(event["fallback_identity"])
        mapping = self.bindings("disabled_optional", p["token_bindings"], used)
        _require(mapping[event["requester_identity"]] == requester
                 and (None if event["fallback_identity"] is None else mapping[event["fallback_identity"]]) == fallback,
                 "disabled component/global token mismatch")
        requester_rule = self.rules[rule["requester"]]
        expected = {
            "environment_sha256": self.environment.sha256, "rule_sha256": canonical_sha256(rule),
            "requester": rule["requester"], "phase": rule["phase"],
            "requester_source_sha256": self.files[requester_rule["root"], requester_rule["path"]]["sha256"],
            "origin": self._origin(rule), "source_sha256": self.files[rule["root"], rule["path"]]["sha256"],
            "loader_kind": "SourceFileLoader", "present": True, "absent_prerequisite": rule["absent_prerequisite"],
            "prerequisite_spec": None, "request_id": name,
            "fallback_attribute": None if fallback is None else rule["fallback_attribute"],
            "create_count": 0, "exec_count": 0, "body_count": 0, "native_create_count": 0,
            "error_type": "ImportError", "error_args": [f"Frozen optional adapter disabled: {name}"],
            "error_name": name, "error_path": self._origin(rule),
        }
        _require(canonical_json_bytes({key: event[key] for key in expected}) == canonical_json_bytes(expected),
                 "disabled optional resolution/error/fallback facts mismatch")
        self.disabled[name] = operation

    def apply(self, value):
        e = exact_fields(value, {"index", "kind", "guard", "lifecycle_event_index", "details"})
        self.index += 1
        _require(_integer(e["index"]) == self.index, "noncontiguous runtime journal")
        self.lci = _integer(e["lifecycle_event_index"])
        _require(self.last_lifecycle <= self.lci < len(self.trace), "runtime lifecycle reference order")
        self.last_lifecycle = self.lci
        self.guard = e["guard"]
        if self.guard is not None:
            exact_fields(self.guard, {"sequence", "role"})
            _integer(self.guard["sequence"])
            _text(self.guard["role"])
        _require(self.guard == self.guards[self.lci][0], "runtime observation guard differs from lifecycle")
        kind = _text(e["kind"])
        _require(kind in _DETAIL_FIELDS, "unknown runtime observation")
        d = exact_fields(e["details"], _DETAIL_FIELDS[kind])
        if kind == "external":
            _require(type(d["event"]) is dict, "external event must be an object")
        failed_fact = (kind == "failure" or kind == "file_call" and d["state"] == "failed"
                       or kind == "external" and d["event"].get("kind") == "failure")
        if kind not in {"source_bound", "phase_completed"} and not failed_fact:
            _require(not self.failed and self.guards[self.lci][1], "runtime success outside original armed setup")
            _require(len(self.phases) < len(self.normal_guards)
                     and self.guard == self.normal_guards[len(self.phases)], "runtime observation transplanted between phase guards")
        _require(self.index == 0 if kind == "source_bound" else self.source is not None,
                 "runtime journal must begin with source binding")
        if kind == "source_bound":
            self.source = {k: self.token(v) for k, v in d.items()}
            _require(len(set(self.source.values())) == 3, "source binding identities must be distinct")
            self.adopted(d["session_token"], roles={"captured_source_session"}, scope=HOST_OWNER, operation="source_custody_bind")
            self.adopted(d["bootstrap_token"], roles={"bootstrap_module"}, scope=HOST_OWNER, operation="source_custody_bind")
        elif kind == "environment_bound":
            self.at_phase(1)
            _require(not self.bound_environment and d["environment_sha256"] == self.environment.sha256,
                     "runtime environment binding mismatch/duplicate")
            for name in ("environment_token", "admission_token"):
                self.adopted(self.token(d[name]), scope=HOST_OWNER)
            self.bound_environment = True
        elif kind == "failure":
            _require(d["source"] in {"source_session", "external_admission", "observer"}, "runtime failure source")
            self.token(d["exception_token"])
            self.failed = True
        elif kind == "file_call":
            self.file_call(d)
        elif kind == "external":
            self.external(d)
        elif kind in {"native_global", "module_replacement", "interpreter_alias"}:
            self.component(kind, d)
        elif kind == "returned_owner":
            self.returned_owner(d)
        elif kind == "runtime_binding":
            self.runtime_binding(d)
        elif kind == "binding_consumed":
            self.at_phase(4)
            _require(self.binding is not None and not self.consumed
                     and self.token(d["binding_token"]) == self.binding["binding_token"], "runtime consumption mismatch")
            current_leaves = [e for e in self.trace[:self.lci + 1]
                              if e["guard"] == self.guard and e["operation"] == "low_level_leaf_entered"]
            _require(not current_leaves, "runtime consumption occurred after upload leaf")
            self.consumed = True
        elif kind == "prepared_storage":
            self.prepared_storage(d)
        elif kind == "input_validated":
            self.at_phase(5)
            _require(self.prepared is not None and not self.validated, "input validation predecessor")
            _require(self.token(d["prepared_token"]) == self.prepared["prepared_token"], "validation prepared substitution")
            _require(d["manifest_sha256"] == self.manifest.sha256
                     and d["content_sha256"] == self.manifest.content_sha256
                     and d["topology_sha256"] == self.manifest.topology_sha256
                     and d["validated_arrays"] == [a.name for a in self.manifest.arrays], "input validation manifest mismatch")
            self.validated = True
        elif kind == "phase_completed":
            self.phase_completed(d)

    def returned_owner(self, d):
        token = self.token(d["object_token"])
        producer = d["producer"]
        if producer == "file_create":
            sequence = _integer(d["call_sequence"])
            _require(sequence < len(self.calls) and d["leaf_event_index"] is None
                     and sequence not in self.returned_files, "returned file reference mismatch")
            call = self.calls[sequence]
            _require(call["state"] == "started" and call["operation"] == "create"
                     and self.delegate_stack and self.delegate_stack[-1] == sequence, "returned file outside create interval")
            self.returned_files[sequence] = token
        else:
            self.at_phase(3)
            _require(producer in {"runtime_target", "runtime_stream"} and producer not in self.returned_runtime
                     and d["call_sequence"] is None, "runtime return kind/order")
            leaf = _integer(d["leaf_event_index"])
            _require(leaf <= self.lci and self.trace[leaf]["operation"] == "low_level_leaf_entered"
                     and self.trace[leaf]["guard"] == self.guard, "runtime return has no actual indexed leaf")
            expected = "indexed_runtime_stream" if producer == "runtime_stream" else (
                "indexed_runtime_import" if self.lane == "cuda" else "indexed_warp_init")
            _require(self.leaf_operations[leaf] == expected, "runtime return references unrelated leaf")
            self.returned_runtime[producer] = token

    def runtime_binding(self, d):
        self.at_phase(3)
        _require(self.binding is None, "duplicate runtime binding")
        for name in _DETAIL_FIELDS["runtime_binding"] - {"stream_identity"}:
            self.token(d[name])
        _require(d["controller_token"] == self.source["controller_token"], "runtime controller substitution")
        _require(d["source_token"] not in self.source.values()
                 and self.first[d["source_token"]] == self.index, "runtime canonical source is not a later distinct owner")
        materialization_end = self.materialization_disarm
        self.adopted(d["source_token"], roles={"canonical_native_input"}, scope=HOST_OWNER, before=materialization_end)
        self.adopted(d["binding_token"], roles={"runtime_setup_binding"}, scope=HOST_OWNER)
        self.adopted(d["module_token"], roles={"external_import_owner"}, scope=HOST_OWNER)
        if self.lane == "cuda":
            _require(d["runtime_token"] == self.file_owners.get("torch._C"), "Torch runtime root substitution")
            self.adopted(d["runtime_token"], roles={"external_import_owner"}, scope=HOST_OWNER)
            self.adopted(d["target_token"], roles={"torch_target_descriptor"}, scope=HOST_OWNER)
        else:
            self.adopted(d["runtime_token"], roles={"warp_runtime_root"}, scope=HOST_OWNER)
            self.adopted(d["target_token"], roles={"warp_runtime_target"}, scope=self.config.device_ordinal)
        self.adopted(d["stream_token"], roles={"torch_runtime_stream" if self.lane == "cuda" else "warp_runtime_stream"},
                     scope=self.config.device_ordinal)
        _require(d["stream_token"] == self.returned_runtime.get("runtime_stream")
                 and d["target_token"] == self.returned_runtime.get("runtime_target"), "runtime returned owner mismatch")
        module_name = "torch" if self.lane == "cuda" else "warp"
        _require(d["module_token"] == self.file_owners.get(module_name)
                 and self.file_states.get(module_name) == "completed", "runtime module lacks completed admission")
        identity = d["stream_identity"]
        _require(type(identity) is list and len(identity) == 2, "runtime stream identity shape")
        _require(_integer(identity[0]) == self.config.device_ordinal, "runtime stream device mismatch")
        _integer(identity[1])
        _require(len({d[k] for k in ("binding_token", "source_token", "controller_token", "stream_token")}) == 4,
                 "runtime binding owner identities collide")
        self.binding = dict(d)

    def prepared_storage(self, d):
        self.at_phase(4)
        _require(self.consumed and self.prepared is None, "prepared storage predecessor")
        for name in ("prepared_token", "source_token", "ready_event_token"):
            self.token(d[name])
        _require(d["source_token"] == self.binding["source_token"]
                 and len({d["prepared_token"], d["source_token"], d["ready_event_token"]}) == 3,
                 "prepared canonical source/wrapper substitution")
        self.adopted(d["prepared_token"], roles={self.lane + "_a2_prepared_input"}, scope=self.config.device_ordinal)
        self.adopted(d["ready_event_token"], roles={self.lane + "_a2_ready_event"}, scope=self.config.device_ordinal)
        manifest = NativeInputManifest.from_payload(d["manifest"])
        _require(manifest.case_id == self.inputs["case_id"] == self.materialization_case,
                 "prepared manifest differs from trusted/completed case")
        _require(type(d["arrays"]) is list and len(d["arrays"]) == len(manifest.arrays), "prepared storage array count")
        for row, array in zip(d["arrays"], manifest.arrays):
            row = exact_fields(row, {"name", "array_token", "storage_owner_token", "signature"})
            _require(row["name"] == array.name, "prepared storage array name/order")
            token = self.token(row["array_token"])
            self.adopted(token, roles={self.lane + "_a2_device_array"}, scope=self.config.device_ordinal)
            owner = self.token(row["storage_owner_token"], nullable=True)
            _require((owner is None) == (self.lane == "warp"), "storage owner lane mismatch")
            _validate_storage_signature(row["signature"], array, self.lane, self.config.device_ordinal)
        _require(d["storage_sha256"] == canonical_sha256(d["arrays"]), "prepared storage digest mismatch")
        self.prepared, self.manifest = dict(d), manifest

    def phase_completed(self, d):
        _require(not self.failed and len(self.phases) < len(PHASES)
                 and d["phase"] == PHASES[len(self.phases)], "runtime phase completion order")
        index = _integer(d["disarmed_event_index"])
        _require(index == self.lci and self.guard is None and self.trace[index]["operation"] == "normal_disarmed",
                 "phase completion lacks original DISARMED")
        sequence = self.trace[index]["guard"]["sequence"]
        _require(self.trace[index]["guard"]["role"] == "setup" and sequence not in self.phase_guards,
                 "phase completion repeats or substitutes guard")
        _require(len(self.phases) < len(self.normal_guards)
                 and self.trace[index]["guard"] == self.normal_guards[len(self.phases)], "phase completion guard order")
        count = len(self.phases)
        expected = (True, self.bound_environment, True, self.binding is not None,
                    self.prepared is not None, self.validated)[count]
        _require(expected and not self.delegate_stack and all(c["state"] == "completed" for c in self.calls),
                 "phase completed with missing outcome or incomplete native call")
        if count == 2:
            _require(d["materialization_case_id"] == self.inputs["case_id"], "completed case differs from trusted launch")
            self.materialization_case = d["materialization_case_id"]
            self.materialization_disarm = index
        else:
            _require(d["materialization_case_id"] is None, "non-materialization phase invented case")
        self.phases.append(d["phase"])
        self.phase_guards.append(sequence)

    def finish(self):
        _require(set(self.first) == set(self.objects), "runtime table contains unreferenced object")
        for token, index in self.first.items():
            _require(self.objects[token]["first_journal_index"] == index, "runtime token first occurrence mismatch")
        for key, component in self.components.items():
            _require(self.component_seen[key] == (0 if component is None else len(component["events"])),
                     "runtime component event missing from journal")
        native = self.components["native_globals"]
        _require((native is not None) == self.bound_environment, "runtime component/environment initialization mismatch")
        if native is not None:
            _require(len(self.calls) == len(native["calls"]), "runtime file-call coverage mismatch")
            for call, component in zip(self.calls, native["calls"]):
                for name in ("sequence", "parent_sequence", "observer", "operation", "phase", "state"):
                    _require(call[name] == component[name], "runtime call differs from native component")
        _require(self.root["environment_sha256"] == (self.environment.sha256 if self.bound_environment else None),
                 "runtime environment digest does not reflect binding")
        summary = {
            "last_completed_phase": self.phases[-1] if self.phases else None,
            "external_event_count": self.external_count,
            "completed_file_count": sum(v == "completed" for v in self.file_states.values()),
            "successful_guard_sequences": self.phase_guards,
            "binding_observed": self.binding is not None,
            "prepared_storage_observed": self.prepared is not None,
            "input_validation_observed": self.validated,
            "final_execution_readiness_issued": False,
        }
        _require(canonical_json_bytes(self.root["summary"]) == canonical_json_bytes(summary), "runtime summary mismatch")


@dataclass(frozen=True, slots=True)
class RuntimeCustodyRecord:
    """Immutable descriptive bytes, always parsed with independent inputs."""

    _encoded: bytes

    @classmethod
    def from_payload(cls, value, *, environment, trusted_launch_inputs):
        validate_runtime_custody_inputs(trusted_launch_inputs, environment)
        p = exact_fields(value, {"schema_version", "source", "environment_sha256", "configuration", "lane_id",
                                 "observation_cut", "journal", "objects", "components", "lifecycle", "summary"})
        _require(p["schema_version"] == RUNTIME_CUSTODY_VERSION, "runtime custody version")
        _source(p["source"])
        for key in ("source", "configuration", "lane_id"):
            _require(canonical_json_bytes(p[key]) == canonical_json_bytes(trusted_launch_inputs[key]),
                     "runtime custody differs from trusted launch inputs")
        cut = exact_fields(p["observation_cut"], {"journal_length", "lifecycle_event_count"})
        _require(type(p["journal"]) is list and bool(p["journal"]), "runtime custody journal is empty")
        _require(_integer(cut["journal_length"], 1) == len(p["journal"]), "runtime journal cut mismatch")
        replay = _RuntimeReplay(p, environment, trusted_launch_inputs)
        _require(_integer(cut["lifecycle_event_count"], 1) == len(replay.trace), "runtime lifecycle cut mismatch")
        for event in p["journal"]:
            replay.apply(event)
        replay.finish()
        return cls(canonical_json_bytes(p))

    def to_payload(self):
        return json.loads(self._encoded)

    @property
    def encoded(self):
        return self._encoded

    @property
    def sha256(self):
        return hashlib.sha256(self._encoded).hexdigest()


def validate_runtime_custody_attachment(runtime_record, environment, trusted_launch_inputs, custody_artifact):
    """Validate the existing custody artifact before joining its worker prefix."""
    validate_custody_artifact(custody_artifact)
    payload = runtime_record.to_payload() if type(runtime_record) is RuntimeCustodyRecord else runtime_record
    record = RuntimeCustodyRecord.from_payload(payload, environment=environment,
                                              trusted_launch_inputs=trusted_launch_inputs).to_payload()
    _require(record["configuration"] == custody_artifact["supervisor"]["configuration"],
             "runtime attachment supervisor configuration mismatch")
    worker = custody_artifact["worker"]
    if worker is None:
        return "unavailable_worker_trace"
    _require(worker["configuration"] == record["configuration"] and worker["lane_id"] == record["lane_id"],
             "runtime attachment worker identity mismatch")
    trace = record["lifecycle"]["transition_trace"]
    _require(worker["transition_trace"][:len(trace)] == trace, "runtime lifecycle is not an exact worker prefix")
    return "worker_prefix_matched"
