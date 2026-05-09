"""Dependency Analyzer node -- AST-based import extraction.

Reads the generated Python code, parses its AST, and identifies any
third-party (non-stdlib) imports.  Updates ``state["dependencies"]``.
"""

from __future__ import annotations

import ast
import sys
from typing import Any


# Build a set of known stdlib module names for the current Python version.
if sys.version_info >= (3, 10):
    from sys import stdlib_module_names as _STDLIB  # type: ignore[attr-defined]

    STDLIB_MODULES: frozenset[str] = frozenset(_STDLIB)
else:
    import pkgutil

    STDLIB_MODULES = frozenset(
        {m.name for m in pkgutil.iter_modules() if m.ispkg is False}
        | {
            "os",
            "sys",
            "re",
            "json",
            "math",
            "pathlib",
            "typing",
            "collections",
            "functools",
            "itertools",
            "subprocess",
            "hashlib",
            "datetime",
            "dataclasses",
            "abc",
            "io",
            "shutil",
            "tempfile",
            "unittest",
            "logging",
            "argparse",
            "textwrap",
            "copy",
            "enum",
            "operator",
            "contextlib",
            "ast",
            "inspect",
            "dis",
            "csv",
            "http",
            "urllib",
            "email",
            "html",
            "xml",
            "socket",
            "ssl",
            "sqlite3",
            "threading",
            "multiprocessing",
            "asyncio",
            "concurrent",
            "signal",
            "struct",
            "ctypes",
            "time",
            "calendar",
            "random",
            "statistics",
            "decimal",
            "fractions",
            "string",
            "pprint",
            "traceback",
            "warnings",
            "platform",
            "glob",
            "fnmatch",
            "linecache",
            "tokenize",
            "token",
            "keyword",
            "builtins",
            "__future__",
            "importlib",
            "zipfile",
            "tarfile",
            "gzip",
            "bz2",
            "lzma",
            "base64",
            "binascii",
            "pickle",
            "shelve",
            "marshal",
            "dbm",
            "uuid",
            "secrets",
            "hmac",
            "queue",
            "heapq",
            "bisect",
            "array",
            "weakref",
            "types",
            "numbers",
            "cmath",
            "codecs",
            "unicodedata",
            "locale",
            "gettext",
            "pdb",
            "profile",
            "cProfile",
            "timeit",
            "trace",
            "gc",
            "sysconfig",
            "venv",
            "ensurepip",
            "pip",
            "site",
            "code",
            "compileall",
            "py_compile",
        }
    )

# Common mapping of import names to pip package names.
IMPORT_TO_PIP: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "gi": "PyGObject",
    "attr": "attrs",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
    "Crypto": "pycryptodome",
    "jose": "python-jose",
    "magic": "python-magic",
    "lxml": "lxml",
    "dateutil": "python-dateutil",
}


def _extract_imports(code: str) -> set[str]:
    """Return top-level import names from *code* via AST parsing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def dependency_node(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze generated code and extract third-party pip dependencies."""
    code = state.get("generated_code", "")
    if not code:
        return {"dependencies": []}

    raw_imports = _extract_imports(code)
    third_party = raw_imports - STDLIB_MODULES

    # Map import names to pip package names.
    deps = sorted(
        IMPORT_TO_PIP.get(name, name)
        for name in third_party
        if name  # skip empty strings
    )

    return {"dependencies": deps}
