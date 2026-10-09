import json

import numpy as np
import pytest

from conftest import ROOT
from dxftrain.calibrate import temperature
from dxftrain.eval import metrics
from dxftrain.eval.metrics import Pred


def P(gold, label, conf, project="P-a"):
    return Pred("x", project, gold, label, conf)


def test_outcome_definitions():
    nl = {"GUARD_ROOM"}
    preds = [P("A", "A", 0.9), P("A", "B", 0.8), P("unknown", "A", 0.95), P("A", "unknown", 0.99),
             P("GUARD_ROOM", "GUARD_ROOM", 0.99), P("A", "A", 0.3)]
    o = metrics.outcomes(preds, 0.5, nl)
    # correct: #0; wrong: #1 and #2 (confident label on a gold-unknown object); unknown: #3, #4 (not learnable), #5 (below tau)
    assert (o["correct"], o["wrong"], o["unknown"], o["n"]) == (1, 2, 3, 6)
    assert o["coverage"] == round(3 / 6, 4)


def test_tau_wrong0_gives_zero_wrong_and_strict_inequality():
    preds = [P("A", "A", 0.9), P("A", "B", 0.8), P("B", "B", 0.8), P("B", "B", 0.95)]
    t = metrics.tau_wrong0(preds, set())
    assert t == 0.8
    o = metrics.outcomes(preds, t, set())
    assert o["wrong"] == 0 and o["correct"] == 2  # the right 0.8 prediction is also withheld (tie)


def test_unknown_is_never_wrong():
    preds = [P("A", "unknown", 1.0), P("unknown", "unknown", 1.0)]
    assert metrics.outcomes(preds, 0.0, set())["wrong"] == 0


def test_temperature_fit_reduces_nll():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 5, 500)
    scores = rng.normal(0, 1, (500, 5))
    scores[np.arange(500), gold] += 1.0
    scores *= 6.0  # over-confident
    T = temperature.fit(scores, gold)
    assert T > 1.5
    assert temperature.nll(scores, gold, T) < temperature.nll(scores, gold, 1.0)


def test_per_fold_spread_reports_every_project():
    preds = [P("A", "A", 0.9, "P-1"), P("A", "B", 0.9, "P-2"), P("A", "A", 0.9, "P-3")]
    s = metrics.per_fold(preds, 0.5, set())
    assert s["spread"]["wrong_rate"]["folds"] == 3 and s["spread"]["wrong_rate"]["max"] == 1.0


def test_promotion_requires_pass_against_current_active(tmp_path, monkeypatch):
    from dxftrain.registry import registry as reg
    monkeypatch.setattr(reg, "ROOT", tmp_path)
    monkeypatch.setattr(reg, "REG", tmp_path / "models" / "registry.json")
    (tmp_path / "models" / "m1").mkdir(parents=True)
    reg.save({"models": [{"id": "m1", "gate_report": "models/m1/gate_report.json"}], "active": None,
              "previous": None, "history": []})
    gate = {"verdict": "INCONCLUSIVE", "reasons": ["G5"], "checks": {"G4_vs_active[no_colour]": {"pass": True, "detail": "no active model in the registry"}}}
    (tmp_path / "models" / "m1" / "gate_report.json").write_text(json.dumps(gate))
    with pytest.raises(SystemExit):
        reg.promote("m1")
    gate["verdict"] = "PASS"
    (tmp_path / "models" / "m1" / "gate_report.json").write_text(json.dumps(gate))
    reg.promote("m1")
    assert reg.load()["active"] == "m1"
    # a second model whose gate ran with no active model is stale now that m1 is active
    r = reg.load()
    r["models"].append({"id": "m2", "gate_report": "models/m2/gate_report.json"})
    reg.save(r)
    (tmp_path / "models" / "m2").mkdir()
    (tmp_path / "models" / "m2" / "gate_report.json").write_text(json.dumps(gate))
    with pytest.raises(SystemExit):
        reg.promote("m2")
    reg.rollback() if reg.load()["previous"] else None


def test_sheet_must_be_approved(tmp_path):
    from dxftrain.feedback import feedback
    (tmp_path / "status.json").write_text(json.dumps({"status": "labelled", "allowed_labels": ["unknown"]}))
    (tmp_path / "sheet.csv").write_text("object_id,label\nx,unknown\n")
    with pytest.raises(SystemExit):
        feedback.ingest_sheet(tmp_path)


def test_corrections_reject_non_anonymised(tmp_path, monkeypatch):
    from dxftrain.feedback import feedback
    monkeypatch.setattr(feedback, "FB", tmp_path / "fb")
    good = {"file_sha256": "a" * 64, "handle": "1F", "project": "P-0123456789", "category": "PLOT_BOUNDARY",
            "corrected_at": "2026-10-07", "features": {"type": "POLYGON"}, "declared_plot_area_m2": 100.0}
    bad = dict(good, LESSEE_NAME="Someone")
    bad2 = dict(good, project="Acme Ltd")
    bad3 = dict(good, features={"type": "POLYGON", "_layer": "PLOT_BOUNDARY"})
    f = tmp_path / "export.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in (good, bad, bad2, bad3)))
    d = feedback.ingest_corrections(f)
    imp = json.loads((d / "import.json").read_text())
    assert imp["imported"] == 1 and sum(imp["rejected"].values()) == 3
