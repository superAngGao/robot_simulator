"""Read the fixed existing toolchain closure; never run or import a backend.

Use -I -S -B. This is deployment-authoring evidence, not build/admission authority.
The captured worker uses the same production scanner to check complete coverage.
"""

import argparse
import hashlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve(strict=True).parents[1]
    sys.path.insert(0, str(root))
    from optical_worker.cuda_compilation import _toolchain_inventory
    from optical_worker.protocol import canonical_json_bytes

    payload = {"schema_version": "q54.p12.4.cuda-toolchain-inventory.v1", **_toolchain_inventory()}
    if "torch" in sys.modules or "warp" in sys.modules:
        raise RuntimeError("offline toolchain inventory imported a backend")
    encoded = canonical_json_bytes(payload)
    with args.output.open("xb") as handle:
        handle.write(encoded)
    print("sha256", hashlib.sha256(encoded).hexdigest(), "files", len(payload["files"]),
          "roots", len(payload["roots"]), "symlinks", len(payload["symlinks"]), "bytes", len(encoded))


if __name__ == "__main__":
    main()
