"""The one URL. POST /classify (one object) and POST /classify_drawing (every polygon of a DXF).

    MODEL_ID=<registry id, default: active> uvicorn serve.wrapper.app:app --port 8090

Confidence is the calibrated softmax over label log-likelihoods (Stage D), never the model's text. Below the
validation threshold, or for a class marked not learnable, the category is "unknown". The prompt is rendered by
the same code and the same prompt file as training; the served prompt hash is returned with every answer, with
the model's gate verdict (a model that did not PASS Stage E is served only when MODEL_ID names it: trial mode).

/classify_drawing runs the TRAINING extractor (dxftrain.data.extract_objects, same version, same plot rule) and the
same feature steps as make_dataset (nearest side, own-layer redaction), so served features equal trained ones. The
layer of an object is never part of its features or prompt.

Access: when CLASSIFIER_API_KEY is set, /classify and /classify_drawing need it, as "X-API-Key: <key>" or
"Authorization: Bearer <key>" (compared in constant time); without it they answer 401. /health is open. The server
binds wherever uvicorn is told (127.0.0.1 by default): put HTTPS in front of it before exposing it beyond the host.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import Depends, FastAPI, HTTPException, Query, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from dxftrain.calibrate import temperature  # noqa: E402
from dxftrain.data import augment, render_prompt as rp  # noqa: E402
from dxftrain.data.extract_objects import EXTRACTOR_VERSION, extract  # noqa: E402
from dxftrain.registry import registry  # noqa: E402

app = FastAPI(title="dxf object classifier")
STATE: dict = {}
LOCK = threading.Lock()  # one MLX model, one request at a time
MAX_DXF_BYTES = 200 * 1024 * 1024


def require_key(request: Request) -> None:
    """The API key check (a no-op when CLASSIFIER_API_KEY is not set)."""
    want = os.environ.get("CLASSIFIER_API_KEY", "")
    if not want:
        return
    auth = request.headers.get("authorization", "")
    got = request.headers.get("x-api-key") or (auth[7:] if auth.lower().startswith("bearer ") else "")
    if not got or not secrets.compare_digest(got.encode(), want.encode()):
        raise HTTPException(401, "missing or wrong API key", headers={"WWW-Authenticate": "Bearer"})


class Req(BaseModel):
    features: dict
    declared_plot_area_m2: float


def _gate_verdict(model_id: str) -> str:
    report = ROOT / "models" / model_id / "gate_report.json"
    return json.loads(report.read_text())["verdict"] if report.exists() else "NOT_GATED"


def _load() -> None:
    reg = registry.load()
    mid = os.environ.get("MODEL_ID") or reg["active"]
    m = next((x for x in reg["models"] if x["id"] == mid), None)
    if m is None:
        raise RuntimeError("no model: register and promote one, or set MODEL_ID")
    cal = json.loads((ROOT / m["calibration"]).read_text())
    if cal["prompt_sha256"] != rp.prompt_sha256() or m["prompt_sha256"] != rp.prompt_sha256():
        raise RuntimeError("prompt file differs from the one the model was trained/calibrated with")
    from dxftrain.infer.backend import make_scorer
    served = str(ROOT / m["served"]) if m.get("served") else cal["base_model"]
    adapter = None if m.get("served") else cal["adapter"]
    STATE.update(model=m, cal=cal, gate=_gate_verdict(mid), active=mid == reg["active"],
                 scorer=make_scorer(served, adapter_path=adapter, labels=cal["labels"],
                                    backend=cal.get("backend", "mlx")))


@app.on_event("startup")
def startup() -> None:
    _load()


def _info() -> dict:
    cal = STATE["cal"]
    return {"model_id": STATE["model"]["id"], "prompt_sha256": rp.prompt_sha256(), "threshold": cal["tau"],
            "provisional": cal["provisional"], "gate_verdict": STATE["gate"], "active": STATE["active"],
            "backend": cal.get("backend", "mlx"), "extractor_version": EXTRACTOR_VERSION}


def _decide(features: dict, declared: float) -> dict:
    """Calibrated answer for one object's name-free features."""
    cal = STATE["cal"]
    with LOCK:
        s = STATE["scorer"].scores(rp.user_message(features, declared))
    p = temperature.softmax(s[None, :], cal["T"])[0]
    j = int(p.argmax())
    label, conf = cal["labels"][j], float(p[j])
    category = label if (label != "unknown" and label not in cal["not_learnable"] and conf > cal["tau"]) else "unknown"
    return {"category": category, "confidence": round(conf, 6), "top_label": label}


@app.get("/health")
def health() -> dict:
    return {"ok": "scorer" in STATE, **(_info() if "scorer" in STATE else {"prompt_sha256": rp.prompt_sha256()})}


@app.post("/classify", dependencies=[Depends(require_key)])
def classify(req: Req) -> dict:
    if "scorer" not in STATE:
        raise HTTPException(503, "model not loaded")
    f = dict(req.features)
    if any(k.startswith("_") or k == "layer" for k in f):
        raise HTTPException(422, "layer names are never accepted as input")
    return {**_decide(f, req.declared_plot_area_m2), **_info()}


def _served_features(o: dict) -> dict:
    """make_dataset.label_objects' feature steps, without the label: nearest side, own-layer redaction."""
    f = augment.with_side(dict(o["features"]))
    for key in ("text", "dim_text_override"):
        if f.get(key):
            f[key], _ = rp.redact_layer(f[key], o["_layer"])
    return f


def _why_no_objects(meta: dict) -> str:
    if meta.get("declared_plot_area_m2") is None:
        return "no PLAN_INFO with PLOT_AREA_M2: the plot cannot be identified, so no plot-relative features"
    n = meta.get("plot_match_count", 0)
    if n == 0:
        return f"no closed polygon within 2% of the declared plot area {meta['declared_plot_area_m2']:g} m2"
    return f"{n} different polygons match the declared plot area: the plot is ambiguous"


@app.post("/classify_drawing", dependencies=[Depends(require_key)])
async def classify_drawing(request: Request,
                           types: str = Query("POLYGON", description="comma separated: POLYGON, DIMENSION, TEXT"),
                           min_area_m2: float = Query(0.0, ge=0, description="polygons smaller than this are skipped"),
                           max_objects: int = Query(600, ge=1, le=5000),
                           with_colour: bool = Query(True)) -> dict:
    """Body: the DXF file's bytes. Returns one answer per kept object, largest polygons first."""
    if "scorer" not in STATE:
        raise HTTPException(503, "model not loaded")
    body = await request.body()
    if not body:
        raise HTTPException(422, "empty body: send the DXF file's bytes")
    if len(body) > MAX_DXF_BYTES:
        raise HTTPException(413, "DXF larger than 200 MB")
    wanted = {t.strip().upper() for t in types.split(",") if t.strip()}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drawing.dxf"
        path.write_bytes(body)
        try:
            meta, objects = extract(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable drawing is the caller's input error
            raise HTTPException(422, f"DXF could not be read: {type(exc).__name__}: {exc}") from exc
    base = {**_info(), "declared_plot_area_m2": meta.get("declared_plot_area_m2"),
            "plot_resolved_by": meta.get("plot_resolved_by"), "plot_handles": [c["handle"] for c in
                                                                              meta.get("plot_candidates", [])]
            if meta.get("plot_resolved_by") else []}
    if not objects:
        return {**base, "status": "no_plot", "reason": _why_no_objects(meta), "objects": [], "skipped": 0,
                "seconds": round(time.time() - t0, 2)}
    kept = [o for o in objects if o["features"]["type"] in wanted
            and (o["features"]["type"] != "POLYGON" or o["features"].get("area_m2", 0) >= min_area_m2)]
    kept.sort(key=lambda o: -(o["features"].get("area_m2") or 0))
    skipped = max(0, len(kept) - max_objects)
    out = []
    for o in kept[:max_objects]:
        f = _served_features(o)
        if not with_colour:
            f["colour"] = None
        ans = _decide(f, meta["declared_plot_area_m2"])
        out.append({"handle": o["handle"], "type": f["type"], "area_m2": f.get("area_m2"),
                    "inside_plot": f.get("inside_plot"), **ans})
    return {**base, "status": "ok", "reason": "", "objects": out, "skipped": skipped,
            "seconds": round(time.time() - t0, 2)}
