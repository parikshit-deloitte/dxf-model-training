import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def latest_dataset():
    ds = sorted((ROOT / "data" / "datasets").glob("ds-*")) if (ROOT / "data" / "datasets").exists() else []
    return ds[-1] if ds else None
