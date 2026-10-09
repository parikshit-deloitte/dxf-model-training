"""Git commit and library versions stamped into every manifest, run and model card."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
from pathlib import Path

LIBS = ["ezdxf", "shapely", "numpy", "scikit-learn", "mlx", "mlx-lm", "transformers", "trl", "peft", "torch",
        "bitsandbytes"]


def git_commit(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return (sha or "no-commit") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment(root: Path) -> dict:
    libs = {}
    for name in LIBS:
        try:
            libs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return {"git_commit": git_commit(root), "python": platform.python_version(), "platform": platform.platform(),
            "libraries": libs}
