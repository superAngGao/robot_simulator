"""Version-independent encoding for the CPU extraction's AST fingerprints.

Do not hash ast.dump(): it is a debug display whose empty-field defaults
changed between Python 3.12 and 3.14. Keep field names, order, None and lists.
Only empty PEP 695 type_params are omitted because Python 3.10/3.11 lack
that field entirely; nonempty type parameters remain part of the identity.
"""

import ast
import hashlib
import json

AST_ENCODING = "cpu-extraction-structural-json-v1"


def structural_ast(value):
    if isinstance(value, ast.AST):
        return [
            type(value).__name__,
            [
                [name, structural_ast(child)]
                for name, child in ast.iter_fields(value)
                if not (
                    name == "type_params"
                    and child == []
                    and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                )
            ],
        ]
    if isinstance(value, list):
        return [structural_ast(child) for child in value]
    if value is Ellipsis:
        return ["constant", "ellipsis"]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported extraction AST value: {type(value).__name__}")


def structural_ast_sha256(node):
    encoded = json.dumps(
        structural_ast(node), ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
