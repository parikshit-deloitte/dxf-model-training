"""The serving wrapper with a stand-in scorer (no GPU): /classify and /classify_drawing."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import ezdxf
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dxftrain.data import render_prompt as rp  # noqa: E402

LABELS = list(rp.category_names())
FOOT = "BLK_n_LVL_n_BLDG_FOOT_PRINT"


class FakeScorer:
    """Footprint for a 200 m2 polygon, unknown for everything else; records every prompt it saw."""

    def __init__(self) -> None:
        self.labels, self.prompts = LABELS, []

    def scores(self, user: str) -> np.ndarray:
        self.prompts.append(user)
        s = np.full(len(LABELS), -20.0)
        s[LABELS.index(FOOT if "area_m2: 200" in user.splitlines() else "unknown")] = 0.0
        return s


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from serve.wrapper import app as w
    cal = {"T": 1.0, "tau": 0.5, "provisional": True, "labels": LABELS, "not_learnable": []}
    monkeypatch.setattr(w, "STATE", {"model": {"id": "test-model"}, "cal": cal, "gate": "INCONCLUSIVE",
                                     "active": False, "scorer": FakeScorer()})
    return TestClient(w.app), w  # no `with`: the startup hook (which loads the real model) does not run


def dxf_bytes(plan_info: bool = True) -> bytes:
    doc = ezdxf.new(units=6)
    msp = doc.modelspace()  # everything on layer 0: a no-layer drawing
    msp.add_lwpolyline([(0, 0), (40, 0), (40, 25), (0, 25)], close=True)          # plot, 1000 m2
    msp.add_lwpolyline([(10, 5), (30, 5), (30, 15), (10, 15)], close=True)        # building, 200 m2
    msp.add_lwpolyline([(1, 1), (2, 1), (2, 2), (1, 2)], close=True)              # 1 m2: below min_area
    if plan_info:
        msp.add_mtext("PLOT_NO=P-1\\PPLOT_AREA_M2=1000.00\\PLESSEE_NAME=TEST", dxfattribs={"insert": (50, 30)})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode()


def test_classify_drawing_uses_training_features(client):
    c, w = client
    r = c.post("/classify_drawing?min_area_m2=5", content=dxf_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["plot_resolved_by"] == "unique"
    assert body["gate_verdict"] == "INCONCLUSIVE" and body["model_id"] == "test-model"
    cats = {round(o["area_m2"]): o["category"] for o in body["objects"]}
    assert cats == {1000: "unknown", 200: FOOT}                       # the 1 m2 polygon was filtered out
    prompts = w.STATE["scorer"].prompts
    assert all("nearest_plot_box_side:" in p for p in prompts)       # same feature steps as make_dataset
    assert not any("layer" in p.lower() for p in prompts)


def test_classify_drawing_without_plan_info_says_why(client):
    c, _ = client
    body = c.post("/classify_drawing", content=dxf_bytes(plan_info=False)).json()
    assert body["status"] == "no_plot" and "PLOT_AREA_M2" in body["reason"] and body["objects"] == []


def test_classify_drawing_rejects_bad_input(client):
    c, _ = client
    assert c.post("/classify_drawing", content=b"").status_code == 422
    assert c.post("/classify_drawing", content=b"not a dxf at all").status_code == 422


def test_classify_refuses_layer_names(client):
    c, _ = client
    r = c.post("/classify", json={"features": {"type": "POLYGON", "layer": "X"}, "declared_plot_area_m2": 1})
    assert r.status_code == 422
    ok = c.post("/classify", json={"features": {"type": "POLYGON", "area_m2": 200}, "declared_plot_area_m2": 1000})
    assert ok.json()["category"] == FOOT and ok.json()["threshold"] == 0.5


def test_api_key_is_required_when_set(client, monkeypatch):
    c, _ = client
    body = {"features": {"type": "POLYGON", "area_m2": 200}, "declared_plot_area_m2": 1000}
    monkeypatch.setenv("CLASSIFIER_API_KEY", "s3cret-key")
    assert c.post("/classify", json=body).status_code == 401
    assert c.post("/classify", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/classify_drawing", content=dxf_bytes()).status_code == 401
    assert c.post("/classify", json=body, headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert c.post("/classify", json=body, headers={"Authorization": "Bearer s3cret-key"}).status_code == 200
    assert c.get("/health").status_code == 200                                    # health stays open
    monkeypatch.delenv("CLASSIFIER_API_KEY")
    assert c.post("/classify", json=body).status_code == 200                      # no key configured: open (local use)
