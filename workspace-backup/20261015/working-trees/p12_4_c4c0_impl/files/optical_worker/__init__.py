"""CPU-safe contracts for the closed native optical worker.

Importing this package must not import NumPy, Torch, Warp, or any optical
backend.  The bootstrap creates the resource guardian through this package
before it admits CUDA-capable code.
"""

from .lifecycle import (
    UNKNOWN_DEVICE_OWNER,
    DeviceScopeDescriptor,
    NativeWorkerResourceGuardian,
    NativeWorkerSetupLifecycle,
    SetupQuiescenceEvidence,
    TransportTerminationEvidence,
    WorkerSetupProtocolError,
    validate_transport_termination_evidence,
)
from .protocol import (
    CLOSED_WORKER_PROTOCOL_VERSION,
    CLOSED_WORKER_TRUST_MODEL_VERSION,
    ClosedNativeLaneWorkerEvidence,
    ClosedWorkerProtocolError,
    ClosedWorkerRuntimeManifest,
    canonical_json_bytes,
    decode_canonical_message,
    encode_framed_message,
    validate_request,
)

__all__ = [
    "CLOSED_WORKER_PROTOCOL_VERSION",
    "CLOSED_WORKER_TRUST_MODEL_VERSION",
    "UNKNOWN_DEVICE_OWNER",
    "ClosedNativeLaneWorkerEvidence",
    "ClosedWorkerProtocolError",
    "ClosedWorkerRuntimeManifest",
    "DeviceScopeDescriptor",
    "NativeWorkerResourceGuardian",
    "NativeWorkerSetupLifecycle",
    "SetupQuiescenceEvidence",
    "TransportTerminationEvidence",
    "WorkerSetupProtocolError",
    "canonical_json_bytes",
    "decode_canonical_message",
    "encode_framed_message",
    "validate_request",
    "validate_transport_termination_evidence",
]
