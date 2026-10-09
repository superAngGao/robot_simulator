"""Complete v3 codec fixtures with real CPU guards and modeled native facts.

No extension/GPU executes here. The public aggregate parser checks all file,
native-pair and adoption joins; installed execution remains required separately.
"""

import copy
from pathlib import Path

import pytest

from optical_worker.lifecycle import DeviceScopeDescriptor
from optical_worker.runtime_custody import PHASES, RuntimeCustodyRecord
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from tests.unit.optics.test_runtime_custody import baseline, deployment


def reindex(payload):
    first = {}
    indices = {row["index"]: index for index, row in enumerate(payload["journal"])}

    def references(value, index, entries):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_token") or entries and key == "token":
                    if item is not None:
                        first.setdefault(item, index)
                else:
                    references(item, index, entries)
        elif isinstance(value, list):
            for item in value:
                references(item, index, entries)

    for index, row in enumerate(payload["journal"]):
        row["index"] = index
        if row["kind"] == "native_scope" and row["details"]["before_index"] is not None:
            row["details"]["before_index"] = indices[row["details"]["before_index"]]
        references(row["details"], index, row["kind"] == "cuda_extensions_bound")
    for row in payload["objects"]:
        row["first_journal_index"] = first[row["token"]]
    payload["observation_cut"]["journal_length"] = len(payload["journal"])


@pytest.fixture
def full_record(baseline, tmp_path):
    payload, base_environment, inputs, controller = baseline
    ep = base_environment.to_payload()
    for key in ("modules", "absent_probes", "disabled_optional_modules", "publications",
                "native_globals", "module_replacements", "namespaces"):
        ep[key] = []
    ep["roots"].append({"id": "cuda_artifacts", "kind": "site", "path": (tmp_path / "artifacts").as_posix()})
    ep["roots"].sort(key=lambda row: row["id"])
    names = ("torch", "torch._C", *(recipe.name for recipe in CUDA_EXTENSION_RECIPES))
    for name in names:
        root = "site" if name.startswith("torch") else "cuda_artifacts"
        relative = "torch/__init__.py" if name == "torch" else name.replace(".", "/") + ".so"
        ep["files"].append({"root": root, "path": relative, "kind": "python" if name == "torch" else "extension",
                            "size": 1, "sha256": "a" * 64})
        ep["modules"].append({"name": name, "root": root, "path": relative,
                              "phase": "runtime", "is_package": name == "torch"})
    library = {"root": "site", "path": "nvidia/cuda_runtime/lib/libcudart.so.12", "sha256": "b" * 64}
    ep["files"].append({**library, "size": 1, "kind": "native"})
    ep["files"].sort(key=lambda row: (row["root"], row["path"]))
    ep["modules"].sort(key=lambda row: row["name"])
    environment = RuntimeEnvironmentContract.from_payload(ep)
    calls, external_count = [], 0
    roots = {row["id"]: row["path"] for row in ep["roots"]}
    rules = {row["name"]: row for row in ep["modules"]}

    def hold(role, operation=None, *, device=False):
        owner = object()
        if device:
            controller.adopt_returned_device_owner(owner, actual_device_ordinal=1, role=role, operation_id=operation or role)
        else:
            controller.adopt_host_owner(owner, role=role, operation_id=operation or role)
        token = len(payload["objects"]) + 1
        payload["objects"].append({"token": token, "first_journal_index": 0,
                                   "guardian_sequence": controller._guard_events[-1]["details"]["sequence"]})
        return token

    def emit(kind, **details):
        last = controller._guard_events[-1]
        row = {"index": len(payload["journal"]), "kind": kind,
               "guard": None if kind == "phase_completed" else copy.deepcopy(last["guard"]),
               "lifecycle_event_index": last["index"], "details": details}
        payload["journal"].append(row)
        return row["index"]

    def phase(index):
        emit("phase_completed", phase=PHASES[index], disarmed_event_index=controller._guard_events[-1]["index"],
             materialization_case_id=inputs["case_id"] if index == 2 else None)

    def leaf(operation, kind="runtime_import_or_init", identity="frozen_runtime_environment_v2"):
        permit = controller.begin_device_leaf(DeviceScopeDescriptor.exact_target(
            operation_id=operation, leaf_kind=kind, device_ordinal=1,
            scope_source="captured_runtime_admission", runtime_identity=identity))
        controller.enter_device_leaf(permit)
        return controller._guard_events[-1]["index"]

    def scope(operation, leaf_index, binding, before=None):
        return emit("native_scope", site="cuda_artifact_" + operation,
                    edge="before" if before is None else "after", before_index=before,
                    operation_event_index=leaf_index, worker_pid=1234, thread_id=1234,
                    **{key: binding[key] for key in ("module_token", "runtime_token", "target_token", "stream_token", "binding_token")},
                    libraries=[library], state={"initialized": True, "logical_device": 0,
                        "logical_basis": "native_return", "raw_device": 1,
                        "primary_contexts": [{"ordinal": 0, "active": False}, {"ordinal": 1, "active": True}],
                        "target_current_stream": 0, "held_stream": {"ordinal": 1, "handle": 23}})

    def file(name, binding=None):
        nonlocal external_count
        rule = rules[name]
        spec, loader = hold("external_import_owner"), hold("external_import_owner")
        module = None

        def external(operation):
            nonlocal external_count
            emit("external", event={"index": external_count, "operation": operation, "name": name,
                                    "kind": "file_module", "phase": "runtime"},
                 provenance={"origin": str(Path(roots[rule["root"]]) / rule["path"]),
                    "delegate_kind": "SourceFileLoader" if name == "torch" else "ExtensionFileLoader",
                    "spec_token": spec, "loader_token": loader, "module_token": module})
            external_count += 1

        external("file_resolved")
        for operation in ("create", "exec"):
            external(operation + "_started")
            frame = {"sequence": len(calls), "parent_sequence": None, "observer": name,
                     "operation": operation, "phase": "runtime", "state": "started",
                     "module_token": module, "spec_token": spec, "loader_token": loader}
            emit("file_call", **frame)
            if binding is not None:
                leaf_index = leaf(name + "." + operation, identity="python_extension_" + operation + "_v1")
                before = scope(operation, leaf_index, binding)
            if operation == "create":
                module = hold("external_import_owner")
                emit("returned_owner", producer="file_create", call_sequence=frame["sequence"],
                     leaf_event_index=None, object_token=module)
            if binding is not None:
                scope(operation, leaf_index, binding, before)
            emit("file_call", **dict(frame, state="returned", module_token=module))
            if operation == "create":
                external("created")
            emit("file_call", **dict(frame, state="completed", module_token=module))
            calls.append({key: frame[key] for key in ("sequence", "parent_sequence", "observer", "operation", "phase")}
                         | {"absent_names": [], "state": "completed"})
            if operation == "exec":
                external("completed")
        return {"name": name, "module_token": module, "spec_token": spec, "loader_token": loader}

    with controller.segment("setup"):
        pass
    phase(0)
    with controller.segment("setup"):
        emit("environment_bound", environment_sha256=environment.sha256,
             environment_token=hold("environment"), admission_token=hold("admission"))
    phase(1)
    with controller.segment("setup"):
        source = hold("canonical_native_input")
    phase(2)
    with controller.segment("setup"):
        torch_module, runtime = file("torch"), file("torch._C")
        index = leaf("indexed_runtime_import")
        target = hold("torch_target_descriptor")
        emit("returned_owner", producer="runtime_target", call_sequence=None, leaf_event_index=index, object_token=target)
        index = leaf("indexed_runtime_stream", kind="allocation")
        stream = hold("torch_runtime_stream", device=True)
        emit("returned_owner", producer="runtime_stream", call_sequence=None, leaf_event_index=index, object_token=stream)
        binding = {"binding_token": hold("runtime_setup_binding"), "controller_token": 3,
                   "module_token": torch_module["module_token"], "runtime_token": runtime["module_token"],
                   "target_token": target, "stream_token": stream, "source_token": source, "stream_identity": [1, 23]}
        emit("runtime_binding", **binding)
        recipes = []
        for recipe in CUDA_EXTENSION_RECIPES:
            row = file(recipe.name, binding)
            row["entries"] = [{"name": name, "token": hold("cuda_artifact_entry", recipe.name + ".entry." + name)}
                              for name in recipe.entries]
            recipes.append(row)
        emit("cuda_extensions_bound", owner_token=hold("cuda_artifact_set", "cuda_artifacts.bind_set"),
             binding_token=binding["binding_token"], recipes=recipes)
    phase(3)
    payload["environment_sha256"] = environment.sha256
    payload["components"].update(
        native_globals={"schema_version": "q54.p12.4.native-global-observations.v1",
                        "environment_sha256": environment.sha256, "calls": calls, "events": []},
        module_replacements={"schema_version": "q54.p12.4.module-replacement-observations.v1",
                             "environment_sha256": environment.sha256, "events": []})
    payload["lifecycle"] = controller.evidence().to_payload()
    payload["observation_cut"]["lifecycle_event_count"] = len(payload["lifecycle"]["transition_trace"])
    payload["summary"].update(last_completed_phase=PHASES[3], external_event_count=external_count,
                              completed_file_count=5, successful_guard_sequences=[0, 1, 2, 3], binding_observed=True)
    reindex(payload)
    RuntimeCustodyRecord.from_payload(payload, environment=environment, trusted_launch_inputs=inputs)
    return payload, environment, inputs


def test_whole_v3_record_joins_original_guards_files_native_pairs_and_named_adoptions(full_record):
    payload, environment, inputs = full_record
    parsed = RuntimeCustodyRecord.from_payload(payload, environment=environment, trusted_launch_inputs=inputs)
    assert parsed.to_payload() == payload
    assert len([row for row in payload["journal"] if row["kind"] == "native_scope"]) == 12
    assert payload["summary"]["final_execution_readiness_issued"] is False


@pytest.mark.parametrize("attack", ("same_recipe_entry", "cross_recipe_entry", "set_alias", "wrong_frame",
                                  "wrong_leaf", "wrong_guard", "early_return", "late_retention", "missing_pair",
                                  "root_only", "late_adoption"))
def test_whole_v3_parser_rejects_mutations_with_independent_lifecycle_fixed(full_record, attack):
    original, environment, inputs = full_record
    payload = copy.deepcopy(original)
    journal = payload["journal"]
    bound = next(row["details"] for row in journal if row["kind"] == "cuda_extensions_bound")
    before = next(row for row in journal if row["kind"] == "native_scope")
    after = next(row for row in journal if row["kind"] == "native_scope" and row["details"]["edge"] == "after")
    if attack in {"same_recipe_entry", "cross_recipe_entry"}:
        first = bound["recipes"][0]["entries"][0]
        other = bound["recipes"][0]["entries"][1] if attack == "same_recipe_entry" else bound["recipes"][1]["entries"][0]
        first["token"], other["token"] = other["token"], first["token"]
    elif attack == "set_alias":
        bound["owner_token"] = bound["recipes"][0]["entries"][0]["token"]
    elif attack == "wrong_frame":
        frame = journal[before["index"] - 1]
        assert frame["kind"] == "file_call"
        frame["details"]["observer"] = CUDA_EXTENSION_RECIPES[1].name
    elif attack == "wrong_leaf":
        before["details"]["operation_event_index"] = next(
            row["details"]["leaf_event_index"] for row in journal
            if row["kind"] == "returned_owner" and row["details"]["producer"] == "runtime_stream")
    elif attack == "wrong_guard":
        before["guard"]["sequence"] = 2
    elif attack == "root_only":
        token = bound["recipes"][0]["entries"][0]["token"]
        next(row for row in payload["objects"] if row["token"] == token)["guardian_sequence"] = None
    elif attack == "late_adoption":
        next(row for row in journal if row["kind"] == "cuda_extensions_bound")["lifecycle_event_index"] -= 1
    elif attack == "early_return":
        returning = journal[after["index"] + 1]
        assert returning["kind"] == "file_call" and returning["details"]["state"] == "returned"
        journal.remove(returning)
        journal.insert(journal.index(after), returning)
    elif attack == "late_retention":
        returned = journal[after["index"] - 1]
        assert returned["kind"] == "returned_owner"
        journal.remove(returned)
        journal.insert(journal.index(after) + 1, returned)
    else:
        journal.remove(after)
    # The first-use table is descriptive, so recompute it instead of allowing an
    # unrelated stale index to reject the intended ordering/identity attack.
    if attack != "set_alias":
        reindex(payload)
    assert payload["lifecycle"] == original["lifecycle"]
    expected = {
        "same_recipe_entry": "runtime adoption operation mismatch",
        "cross_recipe_entry": "runtime adoption operation mismatch",
        "set_alias": "runtime adoption role mismatch",
        "wrong_frame": "file call resolution owner mismatch",
        "wrong_leaf": "native site differs from actual descriptor",
        "wrong_guard": "runtime observation guard differs from lifecycle",
        "early_return": "CUDA artifact returned before its native context pair closed",
        "late_retention": "CUDA artifact context closed before retained create return",
        "missing_pair": "CUDA artifact returned before its native context pair closed",
        "root_only": "runtime owner has no observed direct adoption",
        "late_adoption": "runtime adoption occurs after use",
    }
    with pytest.raises(ValueError, match=expected[attack]):
        RuntimeCustodyRecord.from_payload(payload, environment=environment, trusted_launch_inputs=inputs)
