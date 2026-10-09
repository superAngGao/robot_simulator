"""CPU failure-order and binding checks; no native extension or GPU execution.

The codec fixture holds real guardian adoptions fixed but uses descriptive file
and scope facts. Full file/scope replay and native acceptance remain separate.
"""

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.native_bootstrap import _ExternalSourceLoader, _RuntimeObservation
from optical_worker.native_setup import _cuda_artifact_create_exact, _cuda_artifact_exec_exact
from optical_worker.runtime_custody import PHASES, RUNTIME_CUSTODY_VERSION, _RuntimeReplay
from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from tests.support.guarded_peer import Peer


@pytest.mark.parametrize("failure", (None, "stream_exit", "device_exit", "journal"))
def test_create_retains_module_before_context_exit_or_journal_failure(failure):
    events, module, error = [], object(), RuntimeError("injected boundary failure")

    @contextmanager
    def scope(name, expected, actual):
        assert actual is expected
        events.append(name + "_enter")
        try:
            yield
        finally:
            # Both exit paths must see the actual returned owner already held.
            assert admission.records["extension"]["module"] is module
            events.append(name + "_exit")
            if failure == name + "_exit":
                raise error

    target, stream, spec = object(), object(), object()
    torch = SimpleNamespace(cuda=SimpleNamespace(
        device=lambda value: scope("device", target, value),
        stream=lambda value: scope("stream", stream, value)))

    def append(kind, details):
        assert admission.records["extension"]["module"] is module
        assert kind == "returned_owner" and details["call_sequence"] == 7
        events.append("retained_return")
        if failure == "journal":
            raise error

    def create(actual):
        assert actual is spec and events == ["device_enter", "stream_enter"]
        return module

    observation = SimpleNamespace(append=append, token=lambda value: 1 if value is module else None)
    admission = SimpleNamespace(records={}, session=SimpleNamespace(runtime_observation=observation))
    loader = _ExternalSourceLoader(admission, "extension", SimpleNamespace(create_module=create), "/unused", None)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            _cuda_artifact_create_exact(torch, target, stream, loader, spec, {"sequence": 7})
        assert caught.value is error
    else:
        assert _cuda_artifact_create_exact(torch, target, stream, loader, spec, {"sequence": 7}) is module
    assert events == ["device_enter", "stream_enter", "retained_return", "stream_exit", "device_exit"]
    assert admission.records["extension"]["module"] is module


def test_exec_uses_exact_target_and_stream_and_restores_after_delegate_failure():
    events, target, stream, module = [], object(), object(), object()
    error = RuntimeError("native exec failed")

    @contextmanager
    def scope(name, expected, actual):
        assert actual is expected
        events.append(name + "_enter")
        try:
            yield
        finally:
            events.append(name + "_exit")

    def execute(actual):
        assert actual is module
        events.append("exec")
        raise error

    torch = SimpleNamespace(cuda=SimpleNamespace(
        device=lambda value: scope("device", target, value),
        stream=lambda value: scope("stream", stream, value)))
    with pytest.raises(RuntimeError) as caught:
        _cuda_artifact_exec_exact(torch, target, stream, SimpleNamespace(exec_module=execute), module)
    assert caught.value is error
    assert events == ["device_enter", "stream_enter", "exec", "stream_exit", "device_exit"]


@pytest.fixture
def binding_facts():
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)

    class Session:
        def bound_controller(self):
            return controller

    observation = _RuntimeObservation(Session())
    tokens = {}

    def hold(role, operation=None):
        owner = object()
        controller.adopt_host_owner(owner, role=role, operation_id=operation or role)
        return observation.token(owner)

    try:
        tokens["wrong_guard"] = hold("cuda_artifact_set", "cuda_artifacts.bind_set")
        for _ in range(3):
            with controller.segment("setup"):
                pass
        with controller.segment("setup"):
            binding = {name + "_token": hold(name) for name in (
                "binding", "controller", "module", "runtime", "target", "stream", "source")}
            binding["stream_identity"] = [1, 23]
            observation.append("runtime_binding", binding)
            tokens["unrelated"] = hold("unrelated_host_owner")
            rows, resolutions, modules, calls, pairs = [], {}, {}, [], {}
            for recipe in CUDA_EXTENSION_RECIPES:
                module, spec, loader = (hold("external_import_owner") for _ in range(3))
                modules[recipe.name], resolutions[recipe.name] = module, (spec, loader)
                for operation in ("create", "exec"):
                    sequence = len(calls)
                    calls.append({"state": "completed", "module_token": module,
                                  "spec_token": spec, "loader_token": loader})
                    pairs[recipe.name, operation] = (sequence, sequence)
                entries = [{"name": name, "token": hold("cuda_artifact_entry", recipe.name + ".entry." + name)}
                           for name in recipe.entries]
                rows.append({"name": recipe.name, "module_token": module, "spec_token": spec,
                             "loader_token": loader, "entries": entries})
            details = {"owner_token": hold("cuda_artifact_set", "cuda_artifacts.bind_set"),
                       "binding_token": binding["binding_token"], "recipes": rows}
            observation.append("cuda_extensions_bound", details)
            observation.freeze_adoptions()
            trace = copy.deepcopy(controller._guard_events)
        replay = object.__new__(_RuntimeReplay)
        replay.root = {"schema_version": RUNTIME_CUSTODY_VERSION}
        replay.phases, replay.lane = list(PHASES[:3]), "cuda"
        replay.binding, replay.cuda_extensions = binding, None
        replay.guard = {"sequence": 3, "role": "setup"}
        replay.delegate_stack, replay.native_pairs = [], []
        replay.objects = {token: {"guardian_sequence": observation.adoptions.get(token)}
                          for token, _ in observation.owners.values()}
        replay.first, replay.index, replay.lci = {}, 1, len(trace) - 1
        replay.adoptions = {e["details"]["sequence"]: e for e in trace if e["operation"] == "owner_adopted"}
        replay.file_states = {recipe.name: "completed" for recipe in CUDA_EXTENSION_RECIPES}
        replay.file_owners, replay.file_resolutions = modules, resolutions
        replay.calls, replay.cuda_artifact_pairs = calls, pairs
        yield replay, details, tokens, observation
    finally:
        controller.terminate(fault="setup_failure", detection_phase="cpu_fixture", emergency_target_synchronize=None)
        peer.close()


def test_binding_joins_named_actual_adoptions_and_collector_keeps_entry_tokens(binding_facts):
    replay, details, _, observation = binding_facts
    replay.cuda_extensions_bound(details)
    assert replay.cuda_extensions == details
    for recipe in details["recipes"]:
        for entry in recipe["entries"]:
            assert observation.first[entry["token"]] == 1
            assert entry["token"] in observation.adoptions


@pytest.mark.parametrize("attack", (
    "same_recipe_swap", "cross_recipe_swap", "unrelated_entry", "unrelated_set", "wrong_guard",
    "root_only", "late", "missing_pair", "unfinished_file", "module_swap", "legacy_v2", "warp", "repeated",
))
def test_binding_rejects_substitution_with_adoption_and_file_facts_fixed(binding_facts, attack):
    replay, original, tokens, _ = binding_facts
    d = copy.deepcopy(original)
    first = d["recipes"][0]["entries"][0]
    if attack in {"same_recipe_swap", "cross_recipe_swap"}:
        second = d["recipes"][0]["entries"][1] if attack == "same_recipe_swap" else d["recipes"][1]["entries"][0]
        first["token"], second["token"] = second["token"], first["token"]
    elif attack == "unrelated_entry":
        first["token"] = tokens["unrelated"]
    elif attack == "unrelated_set":
        d["owner_token"] = tokens["unrelated"]
    elif attack == "wrong_guard":
        d["owner_token"] = tokens["wrong_guard"]
    elif attack == "root_only":
        replay.objects[first["token"]]["guardian_sequence"] = None
    elif attack == "late":
        replay.lci -= 1  # The actual set adoption is now beyond the observed cut.
    elif attack == "missing_pair":
        replay.cuda_artifact_pairs.popitem()
    elif attack == "unfinished_file":
        replay.calls[-1]["state"] = "returned"
    elif attack == "module_swap":
        d["recipes"][0]["module_token"], d["recipes"][1]["module_token"] = (
            d["recipes"][1]["module_token"], d["recipes"][0]["module_token"])
    elif attack == "legacy_v2":
        replay.root["schema_version"] = "q54.p12.4.runtime-custody.v2"
    elif attack == "warp":
        replay.lane = "warp"
    elif attack == "repeated":
        replay.cuda_extensions_bound(original)
    with pytest.raises(ValueError):
        replay.cuda_extensions_bound(d)
