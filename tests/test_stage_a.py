import json
import os
import random
import re
import subprocess

import pytest

from conftest import ROOT, latest_dataset
from dxftrain.data import augment as aug
from dxftrain.data import render_prompt as rp
from dxftrain.dxf import manual
from dxftrain.dxf.mtext import parse_plan_info

DS = latest_dataset()
needs_ds = pytest.mark.skipif(DS is None, reason="no dataset built (data/ is local only)")


def rows(name):
    return [json.loads(l) for l in open(DS / name)]


def test_manual_names_only_no_alias_for_misspellings():
    assert manual.classify_layer("COMPUND_WALL").kind == "near_miss"
    assert manual.label_for("COMPUND_WALL", "DIMENSION") == (None, "near_miss")
    assert manual.classify_layer("COMPOUND_WALL").category == "COMPOUND_WALL"
    assert manual.classify_layer("PLOT_BOUNDRY").kind == "near_miss"
    assert manual.classify_layer("BLK_2_FLR_3_BLT_UP_AREA").category == "BLK_n_FLR_n_BLT_UP_AREA"
    assert manual.label_for("PLOT_BOUNDARY", "TEXT") == (None, "type_mismatch")
    assert manual.label_for("SITE_PLAN", "POLYGON") == ("unknown", "context")


def test_plan_info_parse():
    raw = r"{\Fstandard|c0;PLAN INFO\P\PLESSEE_NAME=Acme\PPLOT_NO=19\PPLOT_AREA_M2=4005.00\P}"
    d = parse_plan_info(raw)
    assert d["PLOT_AREA_M2"] == "4005.00" and d["PLOT_NO"] == "19"


def test_d4_is_a_group_action_and_sides_follow():
    f = {"rel_pos": [0.1, 0.8], "orientation": "horizontal", "value_to_plot_width": 2.0, "value_to_plot_height": 3.0}
    assert aug.transform(f, 0) == f
    g = aug.transform(f, 1)  # rot90 swaps axes
    assert g["orientation"] == "vertical" and g["value_to_plot_width"] == 3.0
    for k in range(8):
        assert all(0 <= x <= 1 for x in aug.transform(f, k)["rel_pos"])
    assert aug.with_side({"rel_pos": [0.05, 0.5]})["nearest_side"] == "left"


def test_system_prompt_is_the_file_and_hash_is_stable():
    assert rp.system_prompt().startswith(rp.PROMPT_TEMPLATE.read_text().split("{CATEGORIES}")[0].strip()[:40])
    assert re.fullmatch(r"[0-9a-f]{64}", rp.prompt_sha256())


def test_no_app_imports():
    src = ROOT / "src"
    for p in src.rglob("*.py"):
        text = p.read_text()
        assert not re.search(r"^\s*(from|import)\s+(app|common|services|dxfkit|rules)\b", text, re.M), p


@needs_ds
def test_no_layer_name_in_any_prompt():
    layer = {}
    for f in (ROOT / "data" / "objects").rglob("*.objects.jsonl"):
        sha = f.name.split(".")[0]
        for l in open(f):
            o = json.loads(l)
            layer[f"{sha}:{o['handle']}"] = o["_layer"]
    checked = 0
    for name in os.listdir(DS):
        if not name.endswith(".jsonl"):
            continue
        for r in rows(name):
            user = r["messages"][1]["content"]
            lay = layer[r["object_id"]]
            free = " ".join(str(r["features"].get(k) or "") for k in ("text", "dim_text_override"))
            if len(lay) >= 3:
                assert not rp.redact_layer(free, lay)[1], (r["object_id"], lay)
            assert "_layer" not in user
            fixed = "\n".join(l for l in user.splitlines() if not l.startswith(("text:", "dimension_text:")))
            words = set(re.findall(r"[a-z]+", fixed.lower()))
            assert not {"pass", "fail", "accepted"} & words
            checked += 1
    assert checked > 0


@needs_ds
def test_split_by_project():
    seen = {}
    for name in os.listdir(DS):
        if name.endswith(".jsonl"):
            split = name.split(".")[0]
            for r in rows(name):
                assert seen.setdefault(r["project"], split) == split, r["project"]


@needs_ds
def test_prompt_hash_on_every_row_and_manifest():
    m = json.loads((DS / "manifest.json").read_text())
    assert m["prompt_sha256"] == rp.prompt_sha256()
    for r in rows("val.no_colour.jsonl")[:200]:
        assert r["prompt_sha256"] == m["prompt_sha256"]
        assert r["messages"][0]["content"] == rp.system_prompt()


@needs_ds
def test_val_and_test_never_augmented_and_no_colour_has_no_colour():
    for s in ("val", "test"):
        for v in ("with_colour", "no_colour"):
            assert all(r["source"] == "real" for r in rows(f"{s}.{v}.jsonl"))
    assert all(r["features"]["colour"] is None for r in rows("train.no_colour.jsonl"))


@needs_ds
def test_dataset_is_read_only():
    with pytest.raises(PermissionError):
        open(DS / "manifest.json", "a").close()


def test_h1_resolves_only_identical_copies():
    from dxftrain.data.extract_objects import resolve_plot
    a = {"handle": "A", "vertex_hash": "x"}
    b = {"handle": "B", "vertex_hash": "x"}
    c = {"handle": "C", "vertex_hash": "y"}
    assert resolve_plot([a], "none") == ([a], "unique")
    assert resolve_plot([a, b], "none") == ([], None)          # strict rule: two candidates are not resolved
    assert resolve_plot([a, b], "H1") == ([a, b], "H1")        # identical outline drawn twice: one plot
    assert resolve_plot([a, c], "H1") == ([], None)            # different shapes: never a guess
    assert resolve_plot([], "H1") == ([], None)
