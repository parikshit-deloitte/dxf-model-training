"""Stage C on an NVIDIA GPU (a 16 GB T4 is enough): QLoRA with transformers + PEFT + bitsandbytes.

    python -m dxftrain.train.train_lora --dataset data/datasets/ds-... --seed 20261007 [--smoke]
    python -m dxftrain.train.train_lora ... --smoke --model Qwen/Qwen2.5-0.5B-Instruct --no-4bit   # CPU check

Plain transformers Trainer, no TRL: TRL's API and kernels change between versions (its fused loss needs Triton), so
the completion-only loss is built here. Each row is tokenised with the chat template; the system + user prompt
(with the assistant header) gets label -100, so only the answer '{"category": ...}<|im_end|>' is trained, as with
mlx_lm's mask_prompt. The mask is checked on a real row before training starts.

Same recipe as configs/qlora_12gb.yaml: 4-bit nf4 base in cuda.compute_dtype (float16 on a T4, which has no
bfloat16), LoRA r=16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down, lr 2e-4 cosine with warm-up, batch 1 x 16
accumulation, up to 3 epochs, validation after each epoch, stop at the first val-loss rise, best epoch saved to
adapter_best/. run.json has the same fields as mlx_train.py's (run_id, history, best_epoch, backend, model, ...), so
Stage D, packaging, Stage E and the registry treat both backends alike.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import yaml

from dxftrain.common.versions import environment
from dxftrain.data import render_prompt as rp
from dxftrain.infer.backend import dtype_kw

ROOT = Path(__file__).resolve().parents[3]
IGNORE = -100


def ids_of(tokenised) -> list[int]:
    """apply_chat_template(tokenize=True) returns a list, or a dict with input_ids (transformers 5)."""
    return list(tokenised["input_ids"] if hasattr(tokenised, "keys") else tokenised)


def encode(tok, messages: list[dict], max_len: int) -> dict:
    """input_ids / attention_mask / labels for one row; labels are -100 on the system + user prompt."""
    prompt = ids_of(tok.apply_chat_template(messages[:2], add_generation_prompt=True, tokenize=True))
    full = ids_of(tok.apply_chat_template(messages, tokenize=True))
    if full[:len(prompt)] != prompt:
        raise ValueError("the chat template does not extend the prompt with the answer: cannot mask the prompt")
    full = full[:max_len]
    labels = ([IGNORE] * len(prompt) + full[len(prompt):])[:max_len]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def epoch_history(log_history: list[dict], n_val: int) -> dict:
    """Trainer log -> the fields mlx_train.py writes: per-epoch val loss, best epoch and loss, the raw log."""
    history = [{"epoch": int(round(h["epoch"])), "val_loss": round(float(h["eval_loss"]), 6), "n_val": n_val}
               for h in log_history if "eval_loss" in h]
    best = min(history, key=lambda h: h["val_loss"]) if history else {"epoch": None, "val_loss": None}
    return {"history": history, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "train_log": [h for h in log_history if "loss" in h]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args(argv)

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    cfg = yaml.safe_load((ROOT / "configs" / "qlora_12gb.yaml").read_text())
    ds = Path(a.dataset)
    manifest = json.loads((ds / "manifest.json").read_text())
    assert manifest["prompt_sha256"] == rp.prompt_sha256(), "prompt file changed since the dataset was built"
    v = cfg["train_variant"]
    train = [json.loads(l) for l in open(ds / f"train.{v}.jsonl")]
    val = [json.loads(l) for l in open(ds / f"val.{v}.jsonl")]
    if a.smoke:
        train, val = train[:20], val[:20]
    from dxftrain.infer.backend import base_model
    model_id = a.model or base_model("cuda")  # configs base_model_hf, or DXFTRAIN_BASE_MODEL
    cuda = torch.cuda.is_available()
    backend = "cuda" if cuda else "cpu"
    run_id = f"{'smoke' if a.smoke else backend}-{dt.datetime.now():%Y%m%d-%H%M%S}-s{a.seed}"
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)

    c = cfg["cuda"]
    dtype = getattr(torch, c["compute_dtype"]) if cuda else torch.float32
    if cuda and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU has no bfloat16 (e.g. a T4): set cuda.compute_dtype and bnb_4bit_compute_dtype "
                         "to float16 in configs/qlora_12gb.yaml")
    four_bit = cuda and not a.no_4bit
    transformers.set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=c["bnb_4bit_quant_type"],
                               bnb_4bit_use_double_quant=c["bnb_4bit_use_double_quant"],
                               bnb_4bit_compute_dtype=getattr(torch, c["bnb_4bit_compute_dtype"])) if four_bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, **dtype_kw(dtype),
                                                 device_map={"": 0} if cuda else None)
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=c["gradient_checkpointing"])
    elif c["gradient_checkpointing"] and cuda:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["targets"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = [encode(tok, r["messages"], cfg["max_length"]) for r in train]
    val_ds = [encode(tok, r["messages"], cfg["max_length"]) for r in val]
    trained = tok.decode([t for t in train_ds[0]["labels"] if t != IGNORE])
    if not trained.startswith('{"category"'):  # the loss must cover only the answer
        raise SystemExit(f"loss mask is not completion-only: {trained[:120]!r}")

    grad_acc = 1 if a.smoke else cfg["grad_accumulation"]
    epochs = 1 if a.smoke else cfg["epochs_max"]
    updates = max(1, math.ceil(len(train_ds) / cfg["batch_size"]) * epochs // grad_acc)
    args = TrainingArguments(
        output_dir=str(run / "ckpt"), seed=a.seed, per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=grad_acc, learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_schedule"], warmup_steps=max(1, int(cfg["warmup_ratio"] * updates)),
        num_train_epochs=epochs, max_steps=1 if a.smoke else -1, eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="steps", logging_steps=10, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=2, optim=c["optim"] if four_bit else "adamw_torch",
        fp16=cuda and dtype is torch.float16, bf16=cuda and dtype is torch.bfloat16,
        gradient_checkpointing=False,  # enabled on the model above (prepare_model_for_kbit_training)
        remove_unused_columns=False, report_to="none", dataloader_pin_memory=cuda)
    meta = {"run_id": run_id, "dataset": ds.name, "dataset_sha256": manifest["dataset_sha256"],
            "prompt_sha256": rp.prompt_sha256(), "seed": a.seed, "variant": v, "model": model_id, "backend": backend,
            "trainer": "transformers Trainer + peft, completion-only labels", "n_train": len(train), "n_val": len(val),
            "compute_dtype": str(dtype).replace("torch.", ""), "load_in_4bit": four_bit, "trainable_params": trainable,
            "optimizer_updates": updates, "config": cfg, "trained_tokens_example": trained, **environment(ROOT)}
    (run / "run.json").write_text(json.dumps(meta, indent=1, default=str))

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()
    if a.smoke and not any("eval_loss" in h for h in trainer.state.log_history):
        trainer.state.log_history.append({**trainer.evaluate(), "epoch": 1.0})
    trainer.model.save_pretrained(str(run / "adapter_best"))  # the best epoch: load_best_model_at_end
    meta.update(epoch_history(trainer.state.log_history, len(val)))
    (run / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    if a.smoke:
        from peft import PeftModel
        PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id, **dtype_kw(torch.float32)),
                                  str(run / "adapter_best"))
        print("smoke: adapter saved and reloaded OK")
    print(run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
