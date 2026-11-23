"""
experiment_log.py - idempotent experiment log directory management.

The decorator maps a config (a ``params`` dict, hashed by ``keys``) to a
stable directory ``root/<func_name>/<hash12>/``.

After a successful run it writes ``.manifest.json`` (status=complete plus
checksum over all result files). A repeated call with the same ``keys``
and without ``overwrite`` is skipped. With ``overwrite=True`` the directory
is cleared and the procedure re-runs.

Reads use ``<func>.read_logs(params)`` / ``<func>.get_log_path(params)``;
writes use the ``log_path`` argument injected by the decorator.
"""

import hashlib
import json
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path

MANIFEST_NAME = ".manifest.json"


def make_hash(params):
    data = json.dumps(params, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_checksum(log_path):
    parts = []
    for p in sorted(log_path.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            rel = p.relative_to(log_path).as_posix()
            parts.append(rel + ":" + _file_sha256(p))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

class Tee:
    """Duplicate writes to several streams (screen + file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def experiment_log(keys=None, param="params", root="logs"):
    keys = tuple(keys)

    def decorator(func):
        root_path = Path(root) if root is not None else None

        def _subset(p):
            missing = [k for k in keys if k not in p]
            if missing:
                raise TypeError(func.__name__ + ": missing hash keys " + str(missing))
            return {k: p[k] for k in keys}

        def get_log_path(p, **kwargs):
            if not isinstance(p, dict):
                raise TypeError(func.__name__ + ": params must be a dict")
            if root_path is not None:
                base = root_path
            else:
                base = Path(p.get("results_dir", "logs"))
            return base / func.__name__ / make_hash(_subset(p))

        def read_logs(p, **kwargs):
            lp = get_log_path(p)
            return lp if lp.is_dir() else None

        def _complete(p):
            lp = get_log_path(p)
            m = lp / MANIFEST_NAME
            if not m.exists():
                return False
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                return False
            return (d.get("status") == "complete"
                    and d.get("checksum") == dir_checksum(lp))
        @wraps(func)
        def wrapper(p, *args, **kwargs):
            if not isinstance(p, dict):
                raise TypeError(func.__name__ + ": first argument must be a params dict")
            overwrite = bool(p.get("overwrite", False))
            lp = get_log_path(p)

            if (not overwrite) and lp.is_dir() and _complete(p):
                return {"reused": True, "log_path": str(lp)}

            if overwrite and lp.is_dir():
                shutil.rmtree(lp)

            lp.mkdir(parents=True, exist_ok=True)
            result = func(p, *args, log_path=lp, **kwargs)

            manifest = {
                "status": "complete",
                "checksum": dir_checksum(lp),
                "finished_at": datetime.now().isoformat(),
            }
            (lp / MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return result

        wrapper.log_keys = keys
        wrapper.get_log_path = get_log_path
        wrapper.read_logs = read_logs
        wrapper.dir_checksum = dir_checksum
        return wrapper

    return decorator


if __name__ == "__main__":
    @experiment_log(keys=["dataset", "model", "seed"], root="logs")
    def train_model(params, log_path=None):
        (log_path / "training.csv").write_text("hello", encoding="utf-8")

    p = {"dataset": "CIFAR10", "model": "ResNet18", "seed": 42}
    print("path:", train_model.get_log_path(p))
    print("logs:", train_model.read_logs(p))
    print("run1:", train_model(p))
    print("run2 (reused):", train_model(p))
