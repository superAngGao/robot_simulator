"""CUDA compilation continuation records. Parsing never issues a compiled owner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES

from .compilation_contract import (
    CUDA_COMPILATION_VERSION, CUDA_COMPILE_SEQUENCES, CUDA_COMPILE_STEPS,
    CUDA_CONTINUATION_SOURCE_MODULES, CUDA_NATIVE_SITES, CUDA_TERMINAL_SEQUENCE,
    cuda_native_operation, cuda_role_anchor,
)
from .lifecycle import HOST_OWNER
from .protocol import canonical_json_bytes
from .runtime_custody import (
    PHASES, RUNTIME_CUSTODY_VERSION, RuntimeCustodyRecord, _DETAIL_FIELDS,
    _RuntimeReplay, _digest, _integer, _require,
)
from .wait_protocol import exact_fields

_FIELDS = {
    "compile_root_registered": {"root_token"},
    "compile_owner_retained": {"root_token", "owner_token", "slot"},
    "compile_step": {"recipe_id", "operation", "module_token", "spec_token", "loader_token",
                     "compiled_owner_token", "entry_tokens", "exception_token"},
}
_COMMON = frozenset({"external", "file_call", "native_global", "module_replacement", "interpreter_alias",
                     "returned_owner", "failure", "native_scope"})
_BINDINGS = ("module_token", "spec_token", "loader_token", "compiled_owner_token")


def _references(value):
    result = set()
    if type(value) is dict:
        for key, item in value.items():
            if key in {"event", "manifest", "signature", "local_token"}:
                continue
            if key.endswith("_token") and item is not None:
                result.add(_integer(item, 1))
            else:
                result.update(_references(item))
    elif type(value) is list:
        for item in value:
            result.update(_references(item))
    return result


def _source_trace(trace, *, allowed, complete):
    _require(type(trace) is list, "source trace must be an array")
    state = {}
    for row in trace:
        _require(type(row) is list and len(row) == 2, "source trace row shape")
        operation, name = row
        _require(type(name) is str and name in allowed, "source trace escaped admitted closure")
        if operation == "started":
            _require(name not in state, "source executed twice")
        else:
            _require(operation in {"completed", "failed"} and state.get(name) == "started",
                     "source trace completion order")
        state[name] = operation
    if complete:
        _require(all(value == "completed" for value in state.values()), "incomplete source execution")
    return state


def _file(row, *, directory=None):
    p = exact_fields(row, {"path", "size", "sha256"})
    _require(type(p["path"]) is str, "file path type")
    path = PurePosixPath(p["path"])
    _require(path.is_absolute() and str(path) == p["path"] and ".." not in path.parts, "noncanonical build file path")
    if directory is not None:
        _require(path.parent == PurePosixPath(directory), "generated file escaped recipe directory")
    _integer(p["size"])
    _digest(p["sha256"])
    return p


class _CompilationReplay(_RuntimeReplay):
    def __init__(self, combined, environment, inputs, compilation, upstream):
        super().__init__(combined, environment, inputs)
        self.compilation = compilation
        self.upstream = upstream
        self.cut = len(upstream["journal"])
        self.exported = {row["token"]: row for row in compilation["objects"]}
        self.root_token = None
        self.retained = []
        self.steps = [[] for _ in CUDA_EXTENSION_RECIPES]
        self.recipe_bindings = [dict.fromkeys(_BINDINGS) | {"entry_tokens": []} for _ in CUDA_EXTENSION_RECIPES]
        self.completed = 0
        self.failure_tokens = set()
        self.compile_native = []

    def apply_continuation(self, value):
        kind, details = self._begin_observation(value)
        _require(kind in _COMMON or kind in _FIELDS, "unknown compilation suffix kind")
        d = exact_fields(details, (_DETAIL_FIELDS if kind in _COMMON else _FIELDS)[kind])
        failed_fact = (kind == "failure" or kind == "file_call" and d["state"] == "failed"
                       or kind == "external" and type(d["event"]) is dict and d["event"].get("kind") == "failure"
                       or kind == "compile_step" and d["operation"] == "failed")
        retention = kind == "compile_owner_retained"
        completed = kind == "compile_step" and d["operation"] == "guard_completed"
        terminal = kind == "native_scope" and d["site"] == "terminal_sync"
        if not (failed_fact or retention or completed or terminal):
            _require(not self.failed and self.completed < len(CUDA_EXTENSION_RECIPES)
                     and self.guards[self.lci][1]
                     and self.guard == {"sequence": CUDA_COMPILE_SEQUENCES[self.completed], "role": "setup"},
                     "compiler success outside original active guard")
        for token in _references(d):
            self.token(token)
        if kind == "compile_root_registered":
            _require(self.root_token is None and self.index == self.cut, "compiler root registration order")
            self.root_token = d["root_token"]
            row = self.exported[self.root_token]
            _require(row["origin"] == "continuation" and row["retention_event_index"] is None,
                     "compiler root cannot be an input or indirect owner")
            adoption = self.adopted(self.root_token, roles={"cuda_compile_custody"},
                                    operation="cuda_compile_custody", scope=HOST_OWNER)
            _require(adoption["guard"] == self.guard, "compiler root adopted under wrong guard")
        else:
            _require(self.root_token is not None, "continuation before root custody")
            if kind == "compile_owner_retained":
                token = d["owner_token"]
                _require(d["root_token"] == self.root_token and token != self.root_token
                         and _integer(d["slot"]) == len(self.retained) and token not in self.retained,
                         "compiler retention slot/root/token mismatch")
                row = self.exported[token]
                _require(row["origin"] == "continuation" and row["retention_event_index"] == self.index
                         and row["first_journal_index"] == self.index, "compiler retention is not first occurrence")
                self.retained.append(token)
            elif kind == "compile_step":
                self.step(d)
            elif kind == "failure":
                _require(d["source"] in {"source_session", "external_admission", "observer"}, "failure source")
                self.failed = True
            elif kind == "external":
                self.external(d)
            elif kind == "file_call":
                self.file_call(d)
            elif kind in {"native_global", "module_replacement", "interpreter_alias"}:
                self.component(kind, d)
            elif kind == "returned_owner":
                _require(d["producer"] == "file_create", "compilation cannot repeat runtime initialization")
                self.returned_owner(d)
            elif kind == "native_scope":
                self.compilation_native_scope(d)

    def _role(self, token, recipe, field, entry=None):
        token = self.token(token)
        row = self.exported.get(token)
        _require(row is not None and row["origin"] == "continuation" and token in self.retained,
                 "successful compiled role lacks its retained new object")
        role, operation = cuda_role_anchor(recipe.name, field, entry)
        adopted = self.adopted(token, roles={role}, operation=operation, scope=HOST_OWNER)
        _require(adopted["guard"] == self.guard, "compiled role adopted under another recipe guard")
        _require(self.root["journal"][row["retention_event_index"]]["lifecycle_event_index"] < adopted["index"],
                 "compiled role was not retained before adoption")
        return token

    def step(self, d):
        ids = tuple(r.name for r in CUDA_EXTENSION_RECIPES)
        _require(type(d["recipe_id"]) is str and d["recipe_id"] in ids, "unknown compiler recipe")
        index = ids.index(d["recipe_id"])
        recipe = CUDA_EXTENSION_RECIPES[index]
        _require(index == self.completed, "compiler recipe order/repeated build")
        steps, bindings = self.steps[index], self.recipe_bindings[index]
        operation = d["operation"]
        current = {key: d[key] for key in (*_BINDINGS, "entry_tokens")}
        _require(type(d["entry_tokens"]) is list, "entry bindings must be an array")
        if operation == "failed":
            _require(current == bindings, "failure invented successful bindings")
            token = self.token(d["exception_token"])
            _require(token in self.retained and token not in self.failure_tokens
                     and self.exported[token]["retention_event_index"] >= (steps[-1][1] if steps else self.cut),
                     "failure reused an unrelated earlier exception")
            self.failure_tokens.add(token)
            self.failed = True
            return
        _require(not self.failed and len(steps) < len(CUDA_COMPILE_STEPS)
                 and operation == CUDA_COMPILE_STEPS[len(steps)] and d["exception_token"] is None,
                 "compiled step skipped, duplicated or follows failure")
        expected = dict(bindings)
        if operation == "create_entered":
            for field in ("spec_token", "loader_token"):
                expected[field] = self._role(d[field], recipe, field)
        elif operation == "module_returned":
            expected["module_token"] = self._role(d["module_token"], recipe, "module_token")
        elif operation == "entries_bound":
            entries = d["entry_tokens"]
            _require(len(entries) == len(recipe.entries), "compiled entry count")
            for row, name in zip(entries, recipe.entries):
                exact_fields(row, {"name", "entry_token"})
                _require(row["name"] == name, "compiled entry name/order")
                self._role(row["entry_token"], recipe, "entry_token", name)
            expected["entry_tokens"] = entries
        elif operation == "owner_registered":
            expected["compiled_owner_token"] = self._role(d["compiled_owner_token"], recipe, "compiled_owner_token")
        elif operation == "guard_completed":
            event = self.trace[self.lci]
            _require(self.guard is None and event["operation"] == "normal_disarmed"
                     and event["guard"] == {"sequence": CUDA_COMPILE_SEQUENCES[index], "role": "setup"},
                     "compiler completion lacks matching DISARMED")
            _require(not self.native_pairs and not self.delegate_stack
                     and all(c["state"] == "completed" for c in self.calls)
                     and all(s in {"file_resolved", "completed"} for s in self.file_states.values()),
                     "compiler completed with unfinished native/import call")
            pairs = [(site, edge) for recipe_index, site, edge in self.compile_native if recipe_index == index]
            _require(pairs == [("cuda_compile", "before"), ("cuda_extension_create", "before"),
                               ("cuda_extension_create", "after"), ("cuda_extension_exec", "before"),
                               ("cuda_extension_exec", "after"), ("cuda_compile", "after")],
                     "compiler completion lacks exact native boundary observations")
            self.completed += 1
        _require(current == expected, "compiled successful bindings drift")
        tokens = [v for v in (expected[k] for k in _BINDINGS) if v is not None]
        tokens += [row["entry_token"] for row in expected["entry_tokens"]]
        _require(len(tokens) == len(set(tokens)), "distinct compiled roles share an object")
        bindings.update(expected)
        steps.append((operation, self.index))

    def compilation_native_scope(self, d):
        site, edge = d["site"], d["edge"]
        _require(site in (*CUDA_NATIVE_SITES, "terminal_sync") and edge in {"before", "after"}
                 and not self.failed, "invalid compilation native site/edge")
        operation = _integer(d["operation_event_index"])
        _require(operation <= self.lci and self.trace[operation]["guard"] == self.guard,
                 "native compiler operation guard/cut mismatch")
        identity = (_integer(d["worker_pid"], 1), _integer(d["thread_id"], 1))
        _require(identity[0] == identity[1] and self.native_identity in (None, identity), "compiler native thread changed")
        self.native_identity = identity
        if site == "terminal_sync":
            _require(self.completed == 3 and self.native_emergency[self.lci]
                     and self.guard == {"sequence": CUDA_TERMINAL_SEQUENCE, "role": "emergency"}
                     and self.trace[operation]["operation"] == "emergency_sync_started", "compiler terminal scope")
        else:
            _require(self.completed < 3 and operation in self.leaf_descriptors, "compile observation lacks entered leaf")
            descriptor = self.leaf_descriptors[operation]
            name, runtime = cuda_native_operation(CUDA_EXTENSION_RECIPES[self.completed].name, site)
            _require((descriptor["operation_id"], descriptor["leaf_kind"], descriptor["runtime_identity"])
                     == (name, "compile_or_jit", runtime), "compiler native descriptor differs from recipe")
            latest = max(i for i in self.leaf_descriptors if i <= self.lci)
            _require((site == "cuda_compile" and edge == "after") or operation == latest,
                     "compiler native observation references stale leaf")
        key = site, operation
        if edge == "before":
            _require(d["before_index"] is None and key not in self.native_seen, "duplicate native compiler entry")
            if self.native_pairs:
                _require(len(self.native_pairs) == 1 and self.native_pairs[-1][0][0] == "cuda_compile"
                         and site in {"cuda_extension_create", "cuda_extension_exec"}, "native compiler nesting")
            self.native_seen.add(key)
            self.native_pairs.append((key, self.index))
        else:
            before = _integer(d["before_index"])
            _require(self.native_pairs and self.native_pairs[-1] == (key, before) and before < self.index,
                     "native compiler pair mismatch")
            opening = self.root["journal"][before]
            _require(opening["guard"] == self.guard, "compiler native guard changed")
            for field in ("module_token", "runtime_token", "target_token", "stream_token", "binding_token", "libraries"):
                _require(d[field] == opening["details"][field], "compiler native owners changed")
            _require(d["state"] == opening["details"]["state"], "compiler native state was not restored")
            self.native_pairs.pop()
        self._native_scope_objects(d)
        if site != "terminal_sync":
            self.compile_native.append((self.completed, site, edge))


@dataclass(frozen=True, slots=True)
class CudaCompilationRecord:
    _encoded: bytes

    @classmethod
    def from_payload(cls, value, *, input_record, environment, trusted_launch_inputs, toolchain_sha256):
        _require(type(input_record) is RuntimeCustodyRecord, "exact immutable input record required")
        upstream = RuntimeCustodyRecord.from_payload(input_record.to_payload(), environment=environment,
                                                     trusted_launch_inputs=trusted_launch_inputs).to_payload()
        _require(upstream["schema_version"] == RUNTIME_CUSTODY_VERSION and upstream["lane_id"] == "cuda"
                 and upstream["summary"]["successful_guard_sequences"] == list(range(len(PHASES)))
                 and upstream["summary"]["input_validation_observed"], "compilation input cut incomplete")
        p = exact_fields(value, {"schema_version", "session", "input_evidence_sha256", "source_sha256", "environment_sha256",
                                 "toolchain_sha256", "target_ordinal", "observation_cut", "recipes", "journal", "objects",
                                 "components", "lifecycle", "input_source_trace", "source_trace", "summary"})
        _require(p["schema_version"] == CUDA_COMPILATION_VERSION and p["session"] == upstream["configuration"]["session"]
                 and p["input_evidence_sha256"] == input_record.sha256
                 and p["source_sha256"] == upstream["source"]["snapshot_sha256"]
                 and p["environment_sha256"] == environment.sha256
                 and _digest(p["toolchain_sha256"]) == _digest(toolchain_sha256)
                 and _integer(p["target_ordinal"]) == upstream["configuration"]["device_ordinal"],
                 "compilation upstream identity mismatch")
        cut = exact_fields(p["observation_cut"], {"input_journal_length", "input_lifecycle_event_count", "input_source_trace_length",
                                                "journal_length", "lifecycle_event_count", "source_trace_length"})
        for key, number in cut.items():
            _integer(number)
        j, l = len(upstream["journal"]), len(upstream["lifecycle"]["transition_trace"])
        _require(cut["input_journal_length"] == j and cut["input_lifecycle_event_count"] == l
                 and type(p["journal"]) is list and cut["journal_length"] == j + len(p["journal"])
                 and p["lifecycle"]["transition_trace"][:l] == upstream["lifecycle"]["transition_trace"]
                 and cut["lifecycle_event_count"] == len(p["lifecycle"]["transition_trace"]), "compilation input/lifecycle cut drift")
        _require(type(p["input_source_trace"]) is list and cut["input_source_trace_length"] == len(p["input_source_trace"])
                 and type(p["source_trace"]) is list and cut["source_trace_length"] == len(p["source_trace"])
                 and p["source_trace"][:cut["input_source_trace_length"]] == p["input_source_trace"], "source trace cut mismatch")
        input_names = {row[1] for row in p["input_source_trace"] if type(row) is list and len(row) == 2}
        _source_trace(p["input_source_trace"], allowed=input_names, complete=True)
        source_states = _source_trace(p["source_trace"], allowed=input_names | set(CUDA_CONTINUATION_SOURCE_MODULES), complete=False)
        for key in ("native_globals", "module_replacements", "interpreter_alias"):
            old, new = upstream["components"][key], p["components"][key]
            _require(type(old) is dict and type(new) is dict and set(old) == set(new), "compiler component shape changed")
            for name, item in old.items():
                _require(new[name][:len(item)] == item if name in {"events", "calls"} else new[name] == item,
                         "compiler component input prefix changed")
        _require(type(p["objects"]) is list, "compilation objects must be an array")
        input_objects = {row["token"]: row for row in upstream["objects"]}
        all_objects = dict(input_objects)
        previous = 0
        for raw in p["objects"]:
            row = exact_fields(raw, {"token", "origin", "first_journal_index", "guardian_sequence", "retention_event_index"})
            token = _integer(row["token"], 1)
            _require(token > previous, "compiler object order or duplicate")
            previous = token
            _integer(row["first_journal_index"])
            if row["guardian_sequence"] is not None:
                _integer(row["guardian_sequence"])
            if row["origin"] == "input":
                _require(token in input_objects and row["retention_event_index"] is None
                         and all(row[k] == input_objects[token][k] for k in input_objects[token]), "input token was redefined")
            else:
                _require(row["origin"] == "continuation" and token not in input_objects
                         and row["first_journal_index"] >= j, "new token overlaps input cut")
                if row["retention_event_index"] is not None:
                    _integer(row["retention_event_index"], j)
                all_objects[token] = {key: row[key] for key in ("token", "first_journal_index", "guardian_sequence")}
        combined = dict(upstream, journal=upstream["journal"] + p["journal"], objects=[all_objects[k] for k in sorted(all_objects)],
                        lifecycle=p["lifecycle"], components=p["components"])
        replay = _CompilationReplay(combined, environment, trusted_launch_inputs, p, upstream)
        for event in upstream["journal"]:
            _RuntimeReplay.apply(replay, event)
        _require(not replay.failed and replay.validated and not replay.native_pairs, "input failed before compilation")
        for event in p["journal"]:
            replay.apply_continuation(event)
        replay._finish_common()
        references = set().union(*(_references(e["details"]) for e in p["journal"])) if p["journal"] else set()
        _require(set(replay.exported) == references and replay.root_token is not None, "compiler object projection coverage")
        _require(type(p["recipes"]) is list and len(p["recipes"]) == len(CUDA_EXTENSION_RECIPES), "compiler recipe table coverage")
        for index, (raw, recipe) in enumerate(zip(p["recipes"], CUDA_EXTENSION_RECIPES)):
            row = exact_fields(raw, {"recipe_id", "guard_sequence", "build_directory", "source_files", "generated_files",
                                     "artifact", "entries", "object_bindings"})
            _require(row["recipe_id"] == recipe.name and _integer(row["guard_sequence"]) == CUDA_COMPILE_SEQUENCES[index],
                     "compiler recipe identity/order")
            _require(row["object_bindings"] == {k: replay.recipe_bindings[index][k] for k in _BINDINGS}
                     and row["entries"] == replay.recipe_bindings[index]["entry_tokens"], "derived compiler bindings differ from journal")
            _require(type(row["source_files"]) is list and type(row["generated_files"]) is list, "compiler files must be arrays")
            if replay.steps[index]:
                directory = row["build_directory"]
                _require(type(directory) is str and PurePosixPath(directory).is_absolute()
                         and PurePosixPath(directory).name == recipe.name, "compiler directory recipe binding")
                for file in row["source_files"]:
                    _file(file)
                for file in row["generated_files"]:
                    _file(file, directory=directory)
                if row["artifact"] is not None:
                    file = _file(row["artifact"], directory=directory)
                    _require(PurePosixPath(file["path"]).name == recipe.name + ".so", "compiled artifact name")
                if any(step == "generated" for step, _ in replay.steps[index]):
                    names = [PurePosixPath(f["path"]).name for f in row["generated_files"]]
                    _require(row["artifact"] is not None and len(names) == len(set(names))
                             and {"main.cpp", "cuda.cu", "build.ninja"} <= set(names), "generated compiler files incomplete")
            else:
                _require(row["build_directory"] is None and row["artifact"] is None
                         and not row["source_files"] and not row["generated_files"], "unstarted recipe invented files")
        complete = replay.completed == 3 and not replay.failed
        if complete:
            _require(all(s == "completed" for s in source_states.values())
                     and set(CUDA_CONTINUATION_SOURCE_MODULES) <= set(source_states), "compiled source closure incomplete")
        expected = {"completed_recipes": replay.completed, "failed": replay.failed,
                    "boundary": "compiled_cuda" if complete else "incomplete"}
        _require(canonical_json_bytes(p["summary"]) == canonical_json_bytes(expected), "compiler summary differs from replay")
        return cls(canonical_json_bytes(p))

    def to_payload(self):
        return json.loads(self._encoded)

    @property
    def encoded(self):
        return self._encoded

    @property
    def sha256(self):
        return hashlib.sha256(self._encoded).hexdigest()
