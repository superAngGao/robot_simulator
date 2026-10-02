"""Frozen canonical sequence identities, reproduced in robot_sim/C2c3.

These identify descriptive CPU inputs, never native execution capability.
Each artifact is exactly one canonical JSON record plus one LF terminator;
the identity hashes the record bytes (excluding that JSONL framing byte).
"""

from types import MappingProxyType

RESOLVED_RENDER_SEQUENCE_SHA256 = MappingProxyType(
    {
        "asymmetric_synthetic_7x3_v1": "59e28d8b3e218d3079bf32ab535d97f4e8a3557a6819c2aadafb295ec53f3a24",
        "synthetic_dynamic_rigid_960x640_v1": (
            "5d4c2c54e5cae75c65d1e273bf512cb74ea09fb780dd4db500a1a6cedebf8a70"
        ),
        "go2_static_960x640_v1": "bd8378f0a57bd3a95d501728c9e7c9409e4c4a7e02d7dccf94d5892045b1ccdc",
        "go2_static_1920x1080_v1": "d8519b3977b6d0a144884166ffa943747a389ee61f574ac5cb84e1ae9920250e",
    }
)
