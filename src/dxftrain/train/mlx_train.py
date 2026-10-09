"""Stage C on Apple Silicon: LoRA on the MLX 4-bit Qwen2.5-7B-Instruct with completion-only loss.

    python -m dxftrain.train.mlx_train --dataset data/datasets/ds-... --seed 20261007 [--smoke]

All epochs run in ONE process with ONE optimizer, so the warm-up + cosine schedule and the AdamW moments continue
across epochs (restarting mlx_lm.lora per epoch would replay the warm-up and the high-LR start every epoch).
After every epoch: validation loss on all val rows (n printed), the epoch's adapter saved; stop at the first rise
and keep the best epoch's adapter in adapter_best/. --smoke: 20 examples, 1 step, save + reload, then exit.
Writes runs/<run_id>/ with config, data hashes, per-epoch losses and the adapters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from dxftrain.common.versions import environment
from dxftrain.data import render_prompt as rp

ROOT = Path(__file__).resolve().parents[3]


def write_split(rows: list[dict], path: Path) -> str:
    text = "".join(json.dumps({"messages": r["messages"]}) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def schedule_config(cfg: dict, updates_total: int, smoke: bool) -> dict | None:
    """Warm-up then cosine decay over every optimizer UPDATE of the whole run (all epochs)."""
    if smoke:
        return None
    warm = max(1, int(cfg["warmup_ratio"] * updates_total))
    return {"name": "cosine_decay", "warmup": warm, "arguments": [cfg["learning_rate"], max(1, updates_total - warm)]}


def adapter_config(cfg: dict, lora_cfg: dict) -> dict:
    """What mlx_lm.load(adapter_path=...) and mlx_lm.fuse read from adapter_config.json."""
    return {**lora_cfg, "fine_tune_type": "lora", "num_layers": cfg["mlx"]["num_layers"]}


def save_adapter(model, out: Path, config: dict) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=1))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs" / "qlora_12gb.yaml").read_text())
    ds = Path(a.dataset)
    manifest = json.loads((ds / "manifest.json").read_text())
    assert manifest["prompt_sha256"] == rp.prompt_sha256(), "prompt file changed since the dataset was built"

    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

    run_id = f"{'smoke' if a.smoke else 'mlx'}-{dt.datetime.now():%Y%m%d-%H%M%S}-s{a.seed}"
    run = ROOT / "runs" / run_id
    data_dir = run / "data"
    data_dir.mkdir(parents=True)
    v = cfg["train_variant"]
    train_rows = [json.loads(l) for l in open(ds / f"train.{v}.jsonl")]
    val_rows = [json.loads(l) for l in open(ds / f"val.{v}.jsonl")]
    if a.smoke:
        train_rows, val_rows = train_rows[:20], val_rows[:20]
    hashes = {"train": write_split(train_rows, data_dir / "train.jsonl"),
              "valid": write_split(val_rows, data_dir / "valid.jsonl")}

    lc = cfg["lora"]
    grad_acc = 1 if a.smoke else cfg["grad_accumulation"]
    steps_per_epoch = 1 if a.smoke else math.ceil(len(train_rows) / cfg["batch_size"])
    epochs = 1 if a.smoke else cfg["epochs_max"]
    updates_total = max(1, steps_per_epoch * epochs // grad_acc)
    sched = schedule_config(cfg, updates_total, a.smoke)
    lora_cfg = {
        "model": cfg["mlx"]["base_model"], "mask_prompt": True, "batch_size": cfg["batch_size"],
        "grad_accumulation_steps": grad_acc, "learning_rate": cfg["learning_rate"], "lr_schedule": sched,
        "max_seq_length": cfg["max_length"], "grad_checkpoint": cfg["mlx"]["grad_checkpoint"],
        "optimizer": cfg["mlx"]["optimizer"], "seed": a.seed, "steps_per_epoch": steps_per_epoch,
        "epochs_max": epochs, "updates_total": updates_total,
        "lora_parameters": {"rank": lc["r"], "scale": lc["alpha"] / lc["r"], "dropout": lc["dropout"],
                            "keys": [f"self_attn.{k}" for k in ("q_proj", "k_proj", "v_proj", "o_proj")] +
                                    [f"mlp.{k}" for k in ("gate_proj", "up_proj", "down_proj")]},
    }
    ad_cfg = adapter_config(cfg, lora_cfg)
    meta = {"run_id": run_id, "dataset": ds.name, "dataset_sha256": manifest["dataset_sha256"],
            "prompt_sha256": rp.prompt_sha256(), "seed": a.seed, "variant": v, "n_train": len(train_rows),
            "n_val": len(val_rows), "data_hashes": hashes, "backend": "mlx", "trainer": "in-process, continuous schedule",
            "config": cfg, "lora_cfg": lora_cfg, **environment(ROOT)}
    (run / "run.json").write_text(json.dumps(meta, indent=1))

    np.random.seed(a.seed)
    mx.random.seed(a.seed)
    model, tok = load(lora_cfg["model"])
    model.freeze()
    linear_to_lora_layers(model, cfg["mlx"]["num_layers"] if cfg["mlx"]["num_layers"] > 0 else len(model.layers),
                          lora_cfg["lora_parameters"])
    ds_args = SimpleNamespace(data=str(data_dir), train=True, test=False, mask_prompt=True, hf_dataset=False)
    train_set, valid_set, _ = load_dataset(ds_args, tok)
    train_set, valid_set = CacheDataset(train_set), CacheDataset(valid_set)
    lr = build_schedule(sched) if sched else cfg["learning_rate"]
    opt = optim.AdamW(learning_rate=lr)
    targs = TrainingArgs(batch_size=cfg["batch_size"], iters=steps_per_epoch, val_batches=0,
                         steps_per_report=25, steps_per_eval=10 ** 9, steps_per_save=10 ** 9,
                         adapter_file=str(run / "adapter_last" / "adapters.safetensors"),
                         max_seq_length=cfg["max_length"], grad_checkpoint=cfg["mlx"]["grad_checkpoint"],
                         grad_accumulation_steps=grad_acc)
    (run / "adapter_last").mkdir()

    history, best, best_dir = [], math.inf, run / "adapter_best"
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train(model=model, optimizer=opt, train_dataset=train_set, val_dataset=None, args=targs)
        ad = run / f"adapter_ep{ep}"
        save_adapter(model, ad, ad_cfg)
        vl = evaluate(model=model, dataset=valid_set, batch_size=cfg["batch_size"], num_batches=-1,
                      max_seq_length=cfg["max_length"])
        model.train()
        history.append({"epoch": ep, "val_loss": round(float(vl), 6), "n_val": len(val_rows),
                        "lr_end": float(opt.learning_rate.item()), "optimizer_updates": int(opt.step.item()),
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(history[-1]), flush=True)
        meta["history"] = history
        (run / "run.json").write_text(json.dumps(meta, indent=1))
        if vl < best:
            best = vl
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ad, best_dir)
        else:
            print(f"val loss rose at epoch {ep}: stopping; best epoch kept", flush=True)
            break
    meta["best_val_loss"] = best
    meta["best_epoch"] = min(history, key=lambda h: h["val_loss"])["epoch"]
    (run / "run.json").write_text(json.dumps(meta, indent=1))
    if a.smoke:
        load(lora_cfg["model"], adapter_path=str(best_dir))
        print("smoke: adapter saved and reloaded OK")
    print(run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
