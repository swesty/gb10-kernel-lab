"""Build ~16K tokens of real English text from Python standard-library docstrings."""
import importlib, inspect, sys
mods = ["collections", "itertools", "functools", "asyncio", "json", "re", "os", "pathlib", "subprocess", "threading",
        "logging", "argparse", "datetime", "typing", "dataclasses", "unittest", "socket", "http.client", "email",
        "sqlite3", "csv", "decimal", "fractions", "statistics", "random", "heapq", "bisect", "array", "struct",
        "zipfile", "tarfile", "shutil", "tempfile", "glob", "fnmatch", "pickle", "copy", "pprint", "textwrap",
        "string", "difflib", "urllib.request", "xml.etree.ElementTree", "html.parser", "configparser", "hashlib",
        "hmac", "secrets", "uuid", "contextlib", "abc", "enum", "numbers", "math", "operator", "weakref", "inspect",
        "ast", "dis", "tokenize", "traceback", "warnings", "signal", "selectors", "queue", "sched",
        "multiprocessing", "concurrent.futures"]
out = []
for m in mods:
    try:
        mod = importlib.import_module(m)
    except Exception:
        continue
    for name, obj in inspect.getmembers(mod):
        doc = inspect.getdoc(obj)
        if doc and not name.startswith("_"):
            out.append(f"{m}.{name}: {doc}")
text = "\n\n".join(out)[:62000]
open(sys.argv[1], "w").write(text)
print(len(text), "chars")
