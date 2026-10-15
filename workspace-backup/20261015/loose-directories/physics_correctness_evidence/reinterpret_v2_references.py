"""Reinterpret preserved measurements; this does not run a new trajectory."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "/data2/shared/ang.gao/robot_sim_physics_correctness")
from tests.validation.rigid_body.physics_metrics import first_freefall_penetration

root = Path("/data2/shared/ang.gao/robot_sim_physics_correctness")
source = root / "collab/physics-correctness-evidence/v2-oracle-metrics.json"
rows = []
for row in json.loads(source.read_text()):
    if row.get("solver") != "pgs":
        continue
    step, reference = first_freefall_penetration(0.45, 0.42, row["dt"])
    rows.append({
        "dt": row["dt"],
        "iterations": row["iterations"],
        "recorded_mean_z": row["mean_z"],
        "reference_kind": "first_discrete_penetration",
        "reference_step": step,
        "reference_z": reference,
        "reinterpreted_error_m": row["mean_z"] - reference,
    })
print(json.dumps({
    "provenance": "Post-processing only; measured values are from the preserved v2 run, not a fresh simulation.",
    "source": str(source.relative_to(root)),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "reference_code_commit": "70841b9c509a51c70ca0fe6b1f094597e29026ce",
    "results": rows,
}, indent=2))
