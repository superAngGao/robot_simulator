"""Fixed driver's diagnostic projection only; not custody/admission proof."""

import sys
from types import ModuleType, SimpleNamespace

from tests.support.native_runtime_process import _unrecorded_module_observations


def test_unrecorded_projection_observes_exact_holds_without_mutating_registry(monkeypatch):
    original = dict(sys.modules)
    owned, copied = ModuleType("observed"), ModuleType("observed")
    vars(copied).update(vars(owned))
    for key, value in (("c2diag_owned", owned), ("c2diag_copy", copied),
                       ("c2diag_none", None), ("c2diag_recorded", owned)):
        monkeypatch.setitem(sys.modules, key, value)
    admission = SimpleNamespace(baseline=original, records={"c2diag_recorded": {}}, retained=[owned])
    before = dict(sys.modules)
    result = _unrecorded_module_observations(admission)
    assert result["total"] == 3 and result["truncated"] is False
    rows = {row["key"]: row for row in result["modules"]}
    assert rows["c2diag_owned"]["held_in_failure_graph"] is True
    assert rows["c2diag_copy"]["held_in_failure_graph"] is False
    assert rows["c2diag_owned"]["module_name"] == rows["c2diag_copy"]["module_name"] == "observed"
    assert rows["c2diag_none"]["type"] == "NoneType"
    assert "c2diag_recorded" not in rows
    assert dict(sys.modules) == before and admission.retained == [owned]


def test_unrecorded_projection_reports_truncation_without_claiming_completeness(monkeypatch):
    original = dict(sys.modules)
    for index in range(300):
        monkeypatch.setitem(sys.modules, f"c2diag_{index:03}", ModuleType(f"c2diag_{index:03}"))
    admission = SimpleNamespace(baseline=original, records={}, retained=[])
    result = _unrecorded_module_observations(admission)
    assert result["total"] == 300 and result["truncated"] is True
    assert len(result["modules"]) == 256
    assert result["modules"][0]["key"] == "c2diag_000"
    assert result["modules"][-1]["key"] == "c2diag_255"
    assert all(row["held_in_failure_graph"] is False for row in result["modules"])
