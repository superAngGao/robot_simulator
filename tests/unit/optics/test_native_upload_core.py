"""Shared upload mechanics; CPU tests do not prove real GPU setup safety."""

from __future__ import annotations

import ast
import copy
import inspect
import subprocess
from pathlib import Path

import pytest

from optical_worker import native_setup
from optical_worker.lifecycle import DeviceScopeDescriptor, NativeWorkerSetupLifecycle


class _Owners:
    def __init__(self):
        self.owners = []

    def adopt_host_owner(self, owner, **_):
        self.owners.append(owner)

    def require_adopted(self, *owners):
        if any(not any(held is owner for held in self.owners) for owner in owners):
            raise ValueError("owner missing")


def _context():
    lifecycle = _Owners()
    runtime, target, stream = object(), object(), object()
    context = native_setup._create_upload_context(lifecycle, runtime, target, stream)
    return context, (lifecycle, runtime, target, stream)


def test_context_is_adopted_exactly_and_consumed_once():
    context, owners = _context()
    assert owners[0].owners == [context]
    assert native_setup._consume_upload_context(context) == owners
    assert id(context) not in native_setup._UPLOAD_CONTEXTS
    with pytest.raises(ValueError, match="consumed"):
        native_setup._consume_upload_context(context)


def test_context_accepts_exact_nonweakrefable_production_c0_lifecycle():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=1,
        emergency_sync_notifier=lambda _: pytest.fail("CPU-only context synchronized"),
        session_identity="upload_context_unit",
    )
    runtime, target, stream = object(), object(), object()
    context = native_setup._create_upload_context(lifecycle, runtime, target, stream)
    assert native_setup._consume_upload_context(context) == (lifecycle, runtime, target, stream)
    assert id(context) not in native_setup._UPLOAD_CONTEXTS


def test_failed_context_adoption_removes_strong_identity_registration():
    class RejectOwner(_Owners):
        def adopt_host_owner(self, owner, **_):
            self.rejected = owner
            raise RuntimeError("context adoption rejected")

    lifecycle = RejectOwner()
    before = set(native_setup._UPLOAD_CONTEXTS)
    with pytest.raises(RuntimeError, match="adoption rejected"):
        native_setup._create_upload_context(lifecycle, object(), object(), object())
    assert set(native_setup._UPLOAD_CONTEXTS) == before
    with pytest.raises(ValueError, match="unregistered"):
        native_setup._consume_upload_context(lifecycle.rejected)


@pytest.mark.parametrize("field", ["lifecycle", "runtime", "target", "stream"])
def test_same_context_field_rebinding_burns_registration(field):
    context, _ = _context()
    object.__setattr__(context, field, _Owners() if field == "lifecycle" else object())
    with pytest.raises(ValueError, match="changed"):
        native_setup._consume_upload_context(context)
    assert id(context) not in native_setup._UPLOAD_CONTEXTS


def test_copy_or_handmade_context_has_no_registration():
    context, owners = _context()
    for impostor in (copy.copy(context), native_setup._NativeUploadContext(*owners)):
        with pytest.raises(ValueError, match="unregistered"):
            native_setup._consume_upload_context(impostor)
    assert native_setup._consume_upload_context(context) == owners


def test_missing_guardian_hold_is_not_upload_admission():
    context, owners = _context()
    owners[0].owners.clear()
    with pytest.raises(ValueError, match="owner missing"):
        native_setup._consume_upload_context(context)
    with pytest.raises(ValueError, match="unregistered"):
        native_setup._consume_upload_context(context)


@pytest.mark.parametrize("lane", ["cuda", "warp"])
def test_upload_body_matches_approved_pre_extraction_ast(lane):
    """Freeze old mechanics apart from the explicit operation-ID encoding fix."""
    root = Path(__file__).resolve().parents[3]
    old_text = subprocess.run(
        ["git", "show", "956d1fa8d51b85c25ac17825adef17640f0e69b5:optical_worker/native_setup.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    old = next(
        n
        for n in ast.parse(old_text).body
        if isinstance(n, ast.FunctionDef) and n.name == f"prepare_{lane}_native_input_for_closed_worker"
    )
    old_try = next(n for n in old.body if isinstance(n, ast.Try))
    start = next(
        i for i, n in enumerate(old_try.body) if isinstance(n, ast.Import) and n.names[0].name == "numpy"
    )
    old_body = old_try.body[start:]

    # The original C0 code rejected canonical uppercase names (x_world_R),
    # after earlier copies had already been submitted. Normalize only its five
    # exact per-array operation-ID expressions; retain every other AST node.
    class OperationIds(ast.NodeTransformer):
        changed = 0

        def visit_keyword(self, node):
            if node.arg == "operation_id" and isinstance(node.value, ast.JoinedStr):
                pieces = node.value.values
                assert len(pieces) == 2 and isinstance(pieces[0], ast.Constant)
                assert isinstance(pieces[1], ast.FormattedValue)
                assert ast.dump(pieces[1].value) == ast.dump(ast.parse("manifest.name", mode="eval").body)
                assert pieces[1].conversion == -1 and pieces[1].format_spec is None
                prefix = pieces[0].value
                assert prefix in {
                    f"{lane}_{value}_" for value in ("stage", "host_view", "allocate", "upload")
                }
                node.value = ast.Call(
                    func=ast.Name(id="_array_operation_id", ctx=ast.Load()),
                    args=[ast.Constant(prefix[:-1]), pieces[1].value],
                    keywords=[],
                )
                self.changed += 1
            return node

    original = ast.Module(body=old_body, type_ignores=[])
    ids = OperationIds()
    ids.visit(original)
    assert ids.changed == 5
    new_function = getattr(native_setup, f"_upload_{lane}_native_input")
    new = ast.parse(inspect.getsource(new_function)).body[0]
    new_start = next(
        i for i, n in enumerate(new.body) if isinstance(n, ast.Import) and n.names[0].name == "numpy"
    )
    new_body = new.body[new_start:]
    assert ast.dump(ast.Module(body=old_body, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=new_body, type_ignores=[]),
        include_attributes=False,
    )
    # C0's wrapper must call the sole core, not retain an allocation loop.
    wrapper = ast.parse(
        inspect.getsource(
            getattr(
                native_setup,
                f"prepare_{lane}_native_input_for_closed_worker",
            )
        )
    ).body[0]
    assert not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(wrapper))
    assert (
        sum(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == f"_upload_{lane}_native_input"
            for n in ast.walk(wrapper)
        )
        == 1
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "cuda_stage",
        "cuda_host_view",
        "cuda_allocate",
        "cuda_upload",
        "warp_stage",
        "warp_host_view",
        "warp_allocate",
        "warp_upload",
        "validation_host",
        "validation_view",
        "validation_d2h",
    ],
)
def test_array_operation_ids_preserve_case_distinction_and_are_guardian_valid(prefix):
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="warp",
        target_device_ordinal=1,
        emergency_sync_notifier=lambda _: pytest.fail("CPU operation-ID check synchronized"),
        session_identity="array_ids_unit",
    )
    names = ["fixed_x_world_R", "fixed_x_world_r", "replay_x_world_R", "replay_x_world_r", "A" * 1024]
    ids = [native_setup._array_operation_id(prefix, name) for name in names]
    assert len(set(ids)) == len(names)
    assert ids == [native_setup._array_operation_id(prefix, name) for name in names]
    for identifier in ids:
        assert len(identifier) <= 128
        lifecycle.adopt_host_owner(object(), role="array_name_test", operation_id=identifier)
        DeviceScopeDescriptor.exact_target(
            operation_id=identifier,
            leaf_kind="allocation",
            device_ordinal=1,
            scope_source="array_name_test",
            runtime_identity="cpu_validation",
        )


@pytest.mark.parametrize(
    "prefix,name", [("INVALID", "a"), ("a" * 64, "a"), (None, "a"), ("valid", ""), ("valid", None)]
)
def test_invalid_array_operation_ids_fail_before_resource_registration(prefix, name):
    with pytest.raises(ValueError):
        native_setup._array_operation_id(prefix, name)
