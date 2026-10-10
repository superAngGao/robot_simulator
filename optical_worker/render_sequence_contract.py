"""Frozen canonical sequence identities, reproduced in robot_sim/C2c3.

These identify descriptive CPU inputs, never native execution capability.
Each artifact is exactly one canonical JSON record plus one LF terminator;
the identity hashes the record bytes (excluding that JSONL framing byte).
"""

from types import MappingProxyType

RESOLVED_RENDER_SEQUENCE_SHA256 = MappingProxyType(
    {
        "asymmetric_synthetic_7x3_v1": "f38bcc3199e6778d3494d77307af746b8106c80e8689f2f13fdd17d183b7d062",
        "synthetic_dynamic_rigid_960x640_v1": (
            "2d9d52e156330e7ce13676fcc4f31571de01e3804a58499cf1c678543dd1c7fe"
        ),
        "go2_static_960x640_v1": "a3c03781aaebc634dbca24c5bc2c02941c4f802620e7e499917d5d1778a19306",
        "go2_static_1920x1080_v1": "8f7f52c26e1810fadb01e4d1f3f66f30f202288bf61c24bbf8a4650b447f5a88",
    }
)
