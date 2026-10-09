"""Stage F: package a trained adapter.

    python -m dxftrain.package.package --run runs/<id>            # MLX: fuse -> models/<id>/served (4-bit)
    python -m dxftrain.package.package --run runs/<id> --ollama   # also a de-quantised export + Modelfile
    python -m dxftrain.package.package --run runs/<id> --cuda     # PEFT merge_and_unload -> fp16/bf16 (vLLM/GGUF)

The served artefact is the fused model. It is NOT assumed equal to the evaluated (unfused) model: Stage E G3
compares the two on the test set. The Modelfile embeds the exact system prompt and temperature 0; note that
Ollama cannot return label log-likelihoods, so confidence is computed by serve/wrapper (the one URL), not Ollama.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dxftrain.data import render_prompt as rp

ROOT = Path(__file__).resolve().parents[3]


def modelfile(model_dir: Path) -> str:
    sys_prompt = rp.system_prompt().replace('"""', "'''")
    return (f"# prompt_sha256 {rp.prompt_sha256()}\nFROM {model_dir}\n"
            f'SYSTEM """{sys_prompt}"""\nPARAMETER temperature 0\nPARAMETER num_ctx 1024\n'
            'PARAMETER stop "<|im_end|>"\n')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ollama", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    cal = json.loads((run / "calibration.json").read_text())
    out = ROOT / "models" / meta["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    if meta.get("backend") in ("cuda", "cpu") and not a.cuda:
        # Served as trained: the 4-bit base + the PEFT adapter, loaded by the same TorchScorer that Stage D and
        # Stage E used, so there is no separate artefact to check (gate --served same).
        (out / "package.json").write_text(json.dumps({"served": None, "mode": "base+adapter", "backend": meta["backend"],
                                                      "base": meta["model"], "adapter": cal["adapter"],
                                                      "prompt_sha256": rp.prompt_sha256(), "run": meta["run_id"]}, indent=1))
        print("served as base + adapter (no fused artefact)")
        return 0
    if a.cuda:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import yaml
        name = yaml.safe_load((ROOT / "configs" / "qlora_12gb.yaml").read_text())["cuda"]["compute_dtype"]
        dest = out / f"merged_{'fp16' if name == 'float16' else 'bf16'}"
        from dxftrain.infer.backend import dtype_kw
        base = AutoModelForCausalLM.from_pretrained(meta["model"], **dtype_kw(getattr(torch, name)))
        merged = PeftModel.from_pretrained(base, str(run / "adapter_best")).merge_and_unload()
        merged.save_pretrained(dest)
        AutoTokenizer.from_pretrained(meta["model"]).save_pretrained(dest)
        print(dest)
        return 0
    subprocess.run([sys.executable, "-m", "mlx_lm.fuse", "--model", cal["base_model"], "--adapter-path",
                    cal["adapter"], "--save-path", str(out / "served")], check=True)
    if a.ollama:
        subprocess.run([sys.executable, "-m", "mlx_lm.fuse", "--model", cal["base_model"], "--adapter-path",
                        cal["adapter"], "--save-path", str(out / "dequantised"), "--dequantize"], check=True)
        (out / "Modelfile").write_text(modelfile(out / "dequantised"))
    (out / "package.json").write_text(json.dumps({"served": str((out / "served").relative_to(ROOT)),
                                                  "prompt_sha256": rp.prompt_sha256(), "run": meta["run_id"]}, indent=1))
    print(out / "served")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
