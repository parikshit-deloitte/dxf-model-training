"""Which backend runs here (mlx | cuda | cpu), the base model each one uses, and the one scorer factory.

    python -m dxftrain.infer.backend                 # prints the resolved backend (for scripts)
    python -m dxftrain.infer.backend --zero-shot-dir data/datasets/ds-...   # prints that backend's zero-shot run dir

Every stage that scores (B3, D, E, the server) calls make_scorer(); a run, its calibration.json and the registry
record the backend they were made with, so a model is always scored by the backend it was trained on.
  mlx  : Apple Silicon, mlx-lm, MLX 4-bit base (configs/qlora_12gb.yaml mlx.base_model)
  cuda : NVIDIA GPU, transformers + bitsandbytes 4-bit base in cuda.compute_dtype (float16 on a T4)
  cpu  : transformers in float32, for tests with a small model only (never for a real run)
DXFTRAIN_BACKEND (mlx|cuda|cpu) chooses for "auto"; DXFTRAIN_BASE_MODEL replaces the cuda/cpu base model.
"""

from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("mlx", "cuda", "cpu")


@lru_cache(maxsize=1)
def train_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "qlora_12gb.yaml").read_text())


def _has_mlx() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve(name: str | None = "auto") -> str:
    """'auto' -> mlx on Apple Silicon with mlx-lm, else cuda when a GPU is visible; never cpu by default."""
    name = (name or "auto").lower()
    if name == "auto" and os.environ.get("DXFTRAIN_BACKEND"):  # the environment chooses only when asked to
        name = os.environ["DXFTRAIN_BACKEND"].lower()
    if name == "auto":
        if _has_mlx():
            return "mlx"
        if _has_cuda():
            return "cuda"
        raise SystemExit("no backend: install mlx-lm on Apple Silicon or use an NVIDIA GPU with torch "
                         "(set DXFTRAIN_BACKEND=cpu only for tests with a small model)")
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}: use one of {', '.join(BACKENDS)} or auto")
    return name


def base_model(backend: str) -> str:
    """The base model a backend trains and scores with (the same Qwen2.5-7B-Instruct, two packagings).
    DXFTRAIN_BASE_MODEL replaces it for cuda / cpu (e.g. a small model for an end-to-end test)."""
    cfg = train_config()
    if backend == "mlx":
        return cfg["mlx"]["base_model"]
    return os.environ.get("DXFTRAIN_BASE_MODEL") or cfg["base_model_hf"]


def run_prefix(backend: str) -> str:
    """runs/<prefix>-<time>-s<seed>: the directory prefix a training run of this backend uses."""
    return {"mlx": "mlx", "cuda": "cuda", "cpu": "cpu"}[backend]


def zero_shot_dir(dataset: Path, backend: str) -> Path:
    """The zero-shot run of a dataset; MLX keeps the original name so existing cached scores are reused."""
    name = f"zeroshot-{Path(dataset).name}" + ("" if backend == "mlx" else f"-{backend}")
    return ROOT / "runs" / name


def dtype_kw(dtype) -> dict:
    """from_pretrained's dtype argument: `dtype` from transformers 4.56 on, `torch_dtype` before."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def make_scorer(model_path: str, adapter_path: str | None = None, labels: list[str] | None = None,
                backend: str = "mlx"):
    """A label-likelihood scorer with .labels and .scores(user_message) -> np.ndarray, for this backend.

    A relative adapter path (as runs and calibration.json store it) is resolved against the repository root, so the
    server and the stages work from any working directory."""
    if adapter_path and not Path(adapter_path).is_absolute():
        adapter_path = str(ROOT / adapter_path)
    if backend == "mlx":
        from dxftrain.infer.scorer import MLXScorer
        return MLXScorer(model_path, adapter_path=adapter_path, labels=labels)
    from dxftrain.infer.torch_scorer import TorchScorer
    return TorchScorer(model_path, adapter_path=adapter_path, labels=labels, device=backend)


def main(argv: list[str]) -> int:
    backend = resolve("auto")
    if argv[:1] == ["--zero-shot-dir"]:
        print(zero_shot_dir(Path(argv[1]), backend).relative_to(ROOT))
    else:
        print(backend)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
