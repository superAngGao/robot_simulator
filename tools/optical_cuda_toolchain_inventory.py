"""Read the fixed deployment toolchain; never import a backend or run tools.

Run with -I -S -B. This inventory belongs to deployment, outside the worker.
"""

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path

_CONDA = "/home/ang.gao/miniforge3/envs/robot_sim"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _hash_file(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest, path.stat().st_size


def _toolchain_tools():
    return {
        "nvcc": _CONDA + "/bin/nvcc", "ninja": _CONDA + "/bin/ninja",
        "cxx": "/usr/bin/x86_64-linux-gnu-g++-13", "cc": "/usr/bin/x86_64-linux-gnu-gcc-13",
        "cc1plus": "/usr/libexec/gcc/x86_64-linux-gnu/13/cc1plus",
        "collect2": "/usr/libexec/gcc/x86_64-linux-gnu/13/collect2",
        "as": "/usr/bin/as", "ld": "/usr/bin/ld", "sh": "/bin/sh",
        "cudafe": _CONDA + "/bin/cudafe++", "ptxas": _CONDA + "/bin/ptxas",
        "fatbinary": _CONDA + "/bin/fatbinary", "nvlink": _CONDA + "/bin/nvlink",
        "cicc": _CONDA + "/nvvm/bin/cicc",
    }


def _toolchain_roots():
    """Fixed deployment coverage; callers cannot choose their own input subset."""
    torch = _CONDA + "/lib/python3.12/site-packages/torch"
    trees = (
        _CONDA + "/include", _CONDA + "/targets/x86_64-linux/include", _CONDA + "/bin/crt",
        _CONDA + "/nvvm/libdevice", torch + "/include", "/usr/include", "/usr/local/include",
        "/usr/lib/gcc/x86_64-linux-gnu/13", "/usr/libexec/gcc/x86_64-linux-gnu/13",
        "/usr/lib/x86_64-linux-gnu/ldscripts", "/etc/ld.so.conf.d",
    )
    libraries = (
        _CONDA + "/lib", _CONDA + "/lib/stubs", _CONDA + "/nvvm/lib64",
        _CONDA + "/targets/x86_64-linux/lib", _CONDA + "/targets/x86_64-linux/lib/stubs",
        torch + "/lib", "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu/libfakeroot", "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-13/targets/x86_64-linux/lib", "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
    )
    rows = [(path, "tree", True) for path in trees]
    rows += [(path, "files", True) for path in libraries + (_CONDA + "/bin",)]
    # Record absent fallback/hardware-capability search roots too. Their later
    # appearance changes the snapshot even if old file bytes remain identical.
    rows += [(path, "files", False) for path in (_CONDA + "/lib64", "/usr/local/lib/x86_64-linux-gnu")]
    rows += [(path + "/glibc-hwcaps", "tree", False) for path in libraries]
    rows += [(path, "file", True) for path in (*_toolchain_tools().values(),
             _CONDA + "/bin/nvcc.profile", "/etc/ld.so.conf", "/etc/ld.so.cache", "/lib64/ld-linux-x86-64.so.2")]
    return tuple(sorted(rows))


def _toolchain_inventory():
    """Read the fixed closure without importing dependencies or executing tools."""
    files, links, resolved, resolving, visited = {}, {}, {}, set(), set()

    def resolve(path):
        path = Path(path)
        _require(path.is_absolute(), "inventory path must be absolute")
        if path in resolved:
            return resolved[path]
        _require(path not in resolving, "compiler input symlink cycle")
        resolving.add(path)
        if path.parent == path:
            result = path
        elif path.name == "..":
            result = resolve(path.parent).parent
        else:
            candidate = resolve(path.parent) / path.name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(candidate)
                links[str(candidate)] = {"path": str(candidate), "target": target}
                result = resolve(candidate.parent / target)
            else:
                result = candidate
        resolving.remove(path)
        resolved[path] = result
        return result

    def visit(path, recursive):
        path = resolve(path)
        before = path.stat()
        if stat.S_ISDIR(before.st_mode):
            key = path, recursive
            if key in visited:
                return
            visited.add(key)
            for child in sorted(path.iterdir()):
                actual = resolve(child)
                if recursive or not actual.is_dir():
                    visit(actual, recursive)
            return
        _require(stat.S_ISREG(before.st_mode), "non-regular compiler input")
        if str(path) not in files:
            digest, size = _hash_file(path)
            after = path.stat()
            _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                     == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                     and size == before.st_size and not path.is_symlink(), "compiler input changed while hashing")
            files[str(path)] = {"path": str(path), "size": size, "sha256": digest}

    roots = []
    for name, kind, required in _toolchain_roots():
        present = os.path.lexists(name)
        _require(present or not required, "required compiler input root missing: " + name)
        actual = resolve(Path(name)) if present else None
        roots.append({"path": name, "kind": kind, "present": present,
                      "resolved": None if actual is None else str(actual)})
        if actual is not None:
            _require(actual.is_file() if kind == "file" else actual.is_dir(), "compiler root kind changed")
            visit(actual, kind == "tree")
    tools = {name: str(resolve(Path(path))) for name, path in _toolchain_tools().items()}
    _require(all(path in files for path in tools.values()), "compiler executable missing from coverage")
    for row in links.values():
        _require(Path(row["path"]).is_symlink() and os.readlink(row["path"]) == row["target"],
                 "compiler input symlink changed during inventory")
    for row in roots:
        _require(os.path.lexists(row["path"]) == row["present"]
                 and (not row["present"] or str(Path(row["path"]).resolve(strict=True)) == row["resolved"]),
                 "compiler root changed during inventory")
    return {"roots": roots, "files": [files[path] for path in sorted(files)],
            "symlinks": [links[path] for path in sorted(links)], "tools": tools}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve(strict=True).parents[1]
    sys.path.insert(0, str(root))
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
