"""Historical v1 parser attacks with an archived input and SYNTHETIC continuation.

No synthetic row is a new native observation or live-owner issuance. The archive
is read without extraction or modification; the input cut is explicitly derived.
Actual collector, fixed-driver and native proof remain separate requirements.
"""

import copy
import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from optical_worker.historical.compilation_contract import (
    CUDA_COMPILATION_LEGACY_VERSION as CUDA_COMPILATION_VERSION,
    CUDA_LEGACY_CONTINUATION_SOURCE_MODULES as CUDA_CONTINUATION_SOURCE_MODULES,
    cuda_role_anchor, cuda_native_operation,
)
from optical_worker.historical.cuda_compilation_evidence import CudaCompilationRecord
from optical_worker.guarded_evidence import replay_lifecycle
from optical_worker.lifecycle import HOST_OWNER, DeviceScopeDescriptor
from optical_worker.runtime_custody import RuntimeCustodyRecord
from optical_worker.runtime_environment import RuntimeEnvironmentContract

pytestmark = pytest.mark.skipif(os.name != "posix", reason="unchanged POSIX deployment paths require a POSIX parser host")


@pytest.fixture(scope="module")
def archive_input():
    location = os.environ.get("ROBOT_SIM_COMPILATION_INPUT_ARCHIVE")
    if location is None:
        pytest.skip("retained input archive required; no compilation proof is inferred")
    path = Path(location)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "d872f90f699fea8a1ea3b6a3aecc2c6a481bf34d4936b9a4de732cbaaef23130"
    base = "c2d3-fc86182-native-scope-v2/test_actual_native_scope_under4/"
    with tarfile.open(path) as archive:
        def read(name):
            return json.load(archive.extractfile(base + name))
        original, inputs, request = read("runtime_custody.json"), read("runtime_custody_inputs.json"), read("request.json")
        source_trace = read("input_validation.json")["repository_trace"]
    repo = Path(__file__).resolve().parents[3]
    environment = RuntimeEnvironmentContract.from_payload(json.loads(
        (repo / "tests/data/runtime_environments" / Path(request["environment_path"]).name).read_bytes()))
    RuntimeCustodyRecord.from_payload(original, environment=environment, trusted_launch_inputs=inputs)
    p = copy.deepcopy(original)
    end = next(e for e in p["journal"] if e["kind"] == "phase_completed" and e["details"]["phase"] == "input_validation")
    p["journal"] = p["journal"][:end["index"] + 1]
    trace = p["lifecycle"]["transition_trace"][:end["lifecycle_event_index"] + 1]
    replay = replay_lifecycle(p["configuration"], "cuda", trace)
    p["lifecycle"].update(record_kind="prefix", transition_trace=trace, guardian_coverage=replay.coverage(), summary=replay.summary())
    p["objects"] = [row for row in p["objects"] if row["first_journal_index"] <= end["index"]]
    p["observation_cut"] = {"journal_length": len(p["journal"]), "lifecycle_event_count": len(trace)}
    record = RuntimeCustodyRecord.from_payload(p, environment=environment, trusted_launch_inputs=inputs)
    return record, environment, inputs, source_trace


def synthetic_prefix(archive_input, *, complete=False):
    upstream, environment, inputs, source_trace = archive_input
    old = upstream.to_payload()
    lifecycle = copy.deepcopy(old["lifecycle"])
    trace = lifecycle["transition_trace"]
    replay = replay_lifecycle(old["configuration"], "cuda", trace)
    guard = {"sequence": 6, "role": "setup"}
    journal, objects = [], []
    offset = len(old["journal"])
    next_token = max(row["token"] for row in old["objects"]) + 1
    input_objects = {row["token"]: row for row in old["objects"]}
    native_template = copy.deepcopy(next(e["details"] for e in reversed(old["journal"]) if e["kind"] == "native_scope"))
    recipe_index, slot = 0, 0
    native_open = {}

    def life(operation, details):
        event = {"index": len(trace), "epoch": replay.epoch, "operation": operation,
                 "guard": None if guard is None else dict(guard), "details": details}
        replay.apply(event)
        trace.append(event)

    def append(kind, details):
        journal.append({"index": offset + len(journal), "kind": kind,
                        "guard": None if guard is None else dict(guard),
                        "lifecycle_event_index": len(trace) - 1, "details": copy.deepcopy(details)})

    def adoption(token, role, operation):
        sequence = len(replay.records)
        life("owner_adopted", {"sequence": sequence, "role": role, "owner_type": "synthetic_cpu_owner",
                               "device_scope": HOST_OWNER, "operation_id": operation})
        next(row for row in objects if row["token"] == token)["guardian_sequence"] = sequence

    def retain(*, field=None, entry=None):
        nonlocal next_token, slot
        token = next_token
        next_token += 1
        index = offset + len(journal)
        objects.append({"token": token, "origin": "continuation", "first_journal_index": index,
                        "guardian_sequence": None, "retention_event_index": index})
        append("compile_owner_retained", {"root_token": root, "owner_token": token, "slot": slot})
        slot += 1
        if field is not None:
            role, operation = cuda_role_anchor(CUDA_EXTENSION_RECIPES[recipe_index].name, field, entry)
            adoption(token, role, operation)
        return token

    def native(site, edge):
        if not complete:
            return
        d = copy.deepcopy(native_template)
        if edge == "before":
            name, identity = cuda_native_operation(CUDA_EXTENSION_RECIPES[recipe_index].name, site)
            descriptor = DeviceScopeDescriptor.exact_target(operation_id=name, leaf_kind="compile_or_jit",
                device_ordinal=old["configuration"]["device_ordinal"], scope_source="closed_worker_allowlist", runtime_identity=identity)
            life("descriptor_observed", {"descriptor": descriptor.to_payload()})
            life("setup_operation_started", {"descriptor_sha256": descriptor.sha256, "operation_id": name, "leaf_kind": "compile_or_jit"})
            life("low_level_leaf_entered", {"descriptor_sha256": descriptor.sha256, "operation_sequence": replay.operation_count})
            native_open[site] = len(trace) - 1, offset + len(journal)
        operation, before = native_open[site]
        d.update(site=site, edge=edge, operation_event_index=operation, before_index=None if edge == "before" else before)
        for field in ("module_token", "runtime_token", "target_token", "stream_token", "binding_token"):
            token = d[field]
            if not any(row["token"] == token for row in objects):
                objects.append(dict(input_objects[token], origin="input", retention_event_index=None))
        append("native_scope", d)

    def step(operation):
        if operation == "exec_completed":
            native("cuda_extension_exec", "before")
            native("cuda_extension_exec", "after")
        append("compile_step", {"recipe_id": CUDA_EXTENSION_RECIPES[recipe_index].name, "operation": operation,
                                 **bindings, "entry_tokens": entries, "exception_token": None})
        if operation == "build_entered":
            native("cuda_compile", "before")
        elif operation == "create_entered":
            native("cuda_extension_create", "before")
        elif operation == "module_returned":
            native("cuda_extension_create", "after")
        elif operation == "owner_registered":
            native("cuda_compile", "after")

    recipes = []
    unrelated = None
    for recipe_index, recipe in enumerate(CUDA_EXTENSION_RECIPES):
        bindings = dict.fromkeys(("module_token", "spec_token", "loader_token", "compiled_owner_token"))
        entries = []
        started = complete or recipe_index == 0
        if started:
            guard = {"sequence": 6 + recipe_index, "role": "setup"}
            life("normal_arm_started", {})
            life("guard_armed", {})
            if recipe_index == 0:
                root = next_token
                next_token += 1
                objects.append({"token": root, "origin": "continuation", "first_journal_index": offset,
                                "guardian_sequence": None, "retention_event_index": None})
                adoption(root, "cuda_compile_custody", "cuda_compile_custody")
                append("compile_root_registered", {"root_token": root})
            step("build_entered")
            step("generated")
            bindings["spec_token"] = retain(field="spec_token")
            bindings["loader_token"] = retain(field="loader_token")
            step("create_entered")
            bindings["module_token"] = retain(field="module_token")
            step("module_returned")
            step("exec_completed")
            for name in recipe.entries:
                entries.append({"name": name, "entry_token": retain(field="entry_token", entry=name)})
            step("entries_bound")
            unrelated = retain()
            if complete:
                bindings["compiled_owner_token"] = retain(field="compiled_owner_token")
                step("owner_registered")
                step("hook_restored")
                life("normal_body_returned", {})
                life("normal_complete_started", {})
                life("normal_disarmed", {})
                guard = None
                step("guard_completed")
        directory = "/synthetic/compiler/" + recipe.name
        def file(name):
            return {"path": directory + "/" + name, "size": 1, "sha256": "a" * 64}
        recipes.append({"recipe_id": recipe.name, "guard_sequence": 6 + recipe_index,
                        "build_directory": directory if started else None,
                        "source_files": [{"path": "/synthetic/source/optics/" + recipe.source_module.__name__.split(".")[-1] + ".py",
                                          "size": 1, "sha256": "a" * 64}] if started else [],
                        "generated_files": [file(name) for name in ("main.cpp", "cuda.cu", "build.ninja")] if started else [],
                        "artifact": file(recipe.name + ".so") if started else None,
                        "entries": copy.deepcopy(entries), "object_bindings": dict(bindings)})
    lifecycle.update(guardian_coverage=replay.coverage(), summary=replay.summary())
    continued_trace = copy.deepcopy(source_trace)
    if complete:
        for name in CUDA_CONTINUATION_SOURCE_MODULES:
            if not any(row[1] == name for row in continued_trace):
                continued_trace.extend([["started", name], ["completed", name]])
    payload = {"schema_version": CUDA_COMPILATION_VERSION, "session": old["configuration"]["session"],
               "input_evidence_sha256": upstream.sha256, "source_sha256": old["source"]["snapshot_sha256"],
               "environment_sha256": environment.sha256, "toolchain_sha256": "f" * 64,
               "target_ordinal": old["configuration"]["device_ordinal"],
               "observation_cut": {"input_journal_length": offset,
                                   "input_lifecycle_event_count": len(old["lifecycle"]["transition_trace"]),
                                   "input_source_trace_length": len(source_trace), "journal_length": offset + len(journal),
                                   "lifecycle_event_count": len(trace), "source_trace_length": len(continued_trace)},
               "recipes": recipes, "journal": journal, "objects": sorted(objects, key=lambda row: row["token"]),
               "components": old["components"], "lifecycle": lifecycle,
               "input_source_trace": source_trace, "source_trace": continued_trace,
               "summary": {"completed_recipes": 3 if complete else 0, "failed": False,
                           "boundary": "compiled_cuda" if complete else "incomplete"}}
    return payload, unrelated


def parse(payload, archive_input):
    record, environment, inputs, input_trace = archive_input
    return CudaCompilationRecord.from_payload(payload, input_record=record, input_source_trace=input_trace, environment=environment,
                                              trusted_launch_inputs=inputs, toolchain_sha256="f" * 64)


def test_production_parser_accepts_descriptive_partial_prefix_only(archive_input):
    payload, _ = synthetic_prefix(archive_input)
    record = parse(payload, archive_input)
    assert record.to_payload()["summary"]["boundary"] == "incomplete"
    assert not hasattr(record, "module")


@pytest.mark.parametrize("mutation", ["spec_loader", "entries", "unrelated", "root_only", "wrong_role", "missing_object", "first", "retention", "session"])
def test_role_swaps_reject_with_upstream_lifecycle_and_retention_fixed(archive_input, mutation):
    payload, unrelated = synthetic_prefix(archive_input)
    original = copy.deepcopy(payload)
    bindings = payload["recipes"][0]["object_bindings"]
    swaps = {}
    if mutation == "spec_loader":
        a, b = bindings["spec_token"], bindings["loader_token"]
        swaps = {a: b, b: a}
    elif mutation == "entries":
        a, b = (row["entry_token"] for row in payload["recipes"][0]["entries"])
        swaps = {a: b, b: a}
    elif mutation == "unrelated":
        swaps = {bindings["module_token"]: unrelated}
    elif mutation == "root_only":
        next(row for row in payload["objects"] if row["token"] == bindings["module_token"])["guardian_sequence"] = None
    elif mutation == "wrong_role":
        module_row = next(row for row in payload["objects"] if row["token"] == bindings["module_token"])
        module_row["guardian_sequence"] = next(row for row in payload["objects"] if row["token"] == bindings["loader_token"])["guardian_sequence"]
    elif mutation == "missing_object":
        payload["objects"].pop()
    elif mutation == "first":
        payload["objects"][1]["first_journal_index"] += 1
    elif mutation == "retention":
        payload["objects"][1]["retention_event_index"] += 1
    else:
        payload["session"] = "0" * 32

    def rewrite(value):
        if type(value) is dict:
            for key, item in value.items():
                if key.endswith("_token") and item in swaps:
                    value[key] = swaps[item]
                elif type(item) in (dict, list):
                    rewrite(item)
        elif type(value) is list:
            for item in value:
                rewrite(item)
    if swaps:
        for event in payload["journal"]:
            if event["kind"] == "compile_step":
                rewrite(event["details"])
        rewrite(payload["recipes"])
    assert payload["lifecycle"] == original["lifecycle"]
    assert [e for e in payload["journal"] if e["kind"] == "compile_owner_retained"] == [e for e in original["journal"] if e["kind"] == "compile_owner_retained"]
    with pytest.raises((ValueError, RuntimeError)):
        parse(payload, archive_input)


def test_full_descriptive_three_recipe_grammar_requires_every_native_pair_and_source_cut(archive_input):
    payload, _ = synthetic_prefix(archive_input, complete=True)
    record = parse(payload, archive_input)
    assert record.to_payload()["summary"]["completed_recipes"] == 3
    assert not hasattr(record, "module")
    # A coherent edited input source prefix cannot replace the separately held cut.
    changed = copy.deepcopy(payload)
    changed["input_source_trace"] = changed["input_source_trace"][2:]
    changed["source_trace"] = changed["source_trace"][2:]
    changed["observation_cut"]["input_source_trace_length"] -= 2
    changed["observation_cut"]["source_trace_length"] -= 2
    with pytest.raises((ValueError, RuntimeError)):
        parse(changed, archive_input)
    for index, event in enumerate(payload["journal"]):
        if event["kind"] == "native_scope" and event["details"]["edge"] == "after":
            changed = copy.deepcopy(payload)
            changed["journal"][index]["details"]["before_index"] += 1
            with pytest.raises((ValueError, RuntimeError)):
                parse(changed, archive_input)
            break
