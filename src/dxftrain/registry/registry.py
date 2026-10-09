"""Model registry: models/registry.json records every packaged model and which one is active.

    python -m dxftrain.registry.registry list
    python -m dxftrain.registry.registry register --run runs/<id> --served models/<id>/served
    python -m dxftrain.registry.registry promote <model_id>     # only with a PASS gate report against the CURRENT active
    python -m dxftrain.registry.registry rollback               # active <- previous; nothing is deleted

Promotion refuses unless models/<id>/gate_report.json says PASS and that report's G4 check was run against the
model that is active now (a gate passed against an older active model, or against none when one exists, is stale).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REG = Path(os.environ.get("DXFTRAIN_REGISTRY") or ROOT / "models" / "registry.json")  # override: tests only


def load() -> dict:
    if REG.exists():
        return json.loads(REG.read_text())
    return {"models": [], "active": None, "previous": None, "history": []}


def save(r: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(r, indent=1))


def active() -> dict | None:
    r = load()
    return next((m for m in r["models"] if m["id"] == r["active"]), None)


def register(run_dir: str, served: str | None) -> str:
    run = ROOT / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)
    meta = json.loads((run / "run.json").read_text())
    cal = json.loads((run / "calibration.json").read_text())
    r = load()
    mid = meta["run_id"]
    if any(m["id"] == mid for m in r["models"]):
        raise SystemExit(f"{mid} already registered")
    r["models"].append({"id": mid, "created": dt.datetime.now().isoformat(), "run_dir": str(run.relative_to(ROOT)),
                        "dataset_sha256": meta["dataset_sha256"], "prompt_sha256": meta["prompt_sha256"],
                        "git_commit": meta.get("git_commit"), "seed": meta["seed"], "backend": cal.get("backend", "mlx"), "base": cal["base_model"],
                        "adapter": cal["adapter"], "served": served, "calibration": str((run / "calibration.json").relative_to(ROOT)),
                        "gate_report": f"models/{mid}/gate_report.json"})
    save(r)
    return mid


def promote(mid: str) -> None:
    r = load()
    m = next((x for x in r["models"] if x["id"] == mid), None)
    if m is None:
        raise SystemExit(f"{mid} is not registered")
    gp = ROOT / m["gate_report"]
    if not gp.exists():
        raise SystemExit("no gate report: run Stage E first")
    g = json.loads(gp.read_text())
    if g["verdict"] != "PASS":
        raise SystemExit(f"gate verdict is {g['verdict']}: not promoted ({g['reasons']})")
    vs = [v["detail"] for k, v in g["checks"].items() if k.startswith("G4")]
    cur = r["active"]
    against_current = all((f"active {cur}:" in d) if cur else ("no active model" in d) for d in vs)
    if not against_current:
        raise SystemExit(f"the gate was not run against the current active model ({cur}): re-run Stage E")
    r["previous"], r["active"] = cur, mid
    r["history"].append({"at": dt.datetime.now().isoformat(), "action": "promote", "model": mid, "previous": cur})
    save(r)


def rollback() -> None:
    r = load()
    if not r["previous"]:
        raise SystemExit("no previous model to roll back to")
    r["active"], r["previous"] = r["previous"], r["active"]
    r["history"].append({"at": dt.datetime.now().isoformat(), "action": "rollback", "model": r["active"]})
    save(r)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        print(json.dumps(load(), indent=1))
    elif cmd == "register":
        args = dict(zip(argv[1::2], argv[2::2]))
        print(register(args["--run"], args.get("--served")))
    elif cmd == "promote":
        promote(argv[1])
        print("active:", load()["active"])
    elif cmd == "rollback":
        rollback()
        print("active:", load()["active"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
