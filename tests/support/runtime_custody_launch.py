"""Trusted sacrificial-launch helpers; excluded from the production closure."""

import json
import subprocess
import sys
from pathlib import Path


def capture_runtime_custody_inputs(request, *, executable=None):
    """Capture upstream facts in a separate cold interpreter, before the worker."""
    selected = {key: request[key] for key in (
        "root", "revision", "lane", "catalog_sha256", "guard_configuration", "case_id",
    )}
    result = subprocess.run(
        [executable or sys.executable, "-I", "-S", "-B", str(Path(__file__).with_name("runtime_custody_preflight.py"))],
        input=json.dumps(selected), capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError("runtime custody CPU preflight failed: " + result.stderr)
    return json.loads(result.stdout)
