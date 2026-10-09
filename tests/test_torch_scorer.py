"""The PyTorch scorer (CUDA backend, run here on CPU with the small Qwen2.5-0.5B-Instruct): its cached, batched,
chunked scores must equal a brute-force computation over every full sequence. Skipped when torch or the small model
is not available (HF_HUB_OFFLINE=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')")."""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dxftrain.data import render_prompt as rp  # noqa: E402

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
USER = ("declared_plot_area_m2: 1600\ntype: POLYGON\ncolour: not available\narea_m2: 412.5\narea_ratio_to_plot: 0.2578\n"
        "vertices: 4\ninside_plot: yes\ntouches_plot_edge: no\nposition_in_plot_box: 0.52, 0.47")


@pytest.fixture(scope="module")
def scorer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from dxftrain.infer.torch_scorer import TorchScorer
    try:
        return TorchScorer(SMALL, device="cpu", chunk=55)
    except OSError:
        pytest.skip(f"{SMALL} not in the local Hugging Face cache")


def brute_force(s, user: str) -> np.ndarray:
    """Each label: run prompt + answer tokens as one sequence, sum the answer tokens' log-probabilities."""
    ids = s._ids([{"role": "system", "content": rp.system_prompt()}, {"role": "user", "content": user}])
    out = []
    with torch.inference_mode():
        for i, n_tok in enumerate(s.cand_len):
            cand = s.cand[i, :n_tok].tolist()
            x = torch.tensor([ids + cand])
            lp = torch.log_softmax(s.model(input_ids=x).logits[0].float(), dim=-1)
            out.append(sum(lp[len(ids) - 1 + j, t].item() for j, t in enumerate(cand)))
    return np.array(out)


def test_cached_batched_scores_equal_brute_force(scorer):
    fast = scorer.scores(USER)
    slow = brute_force(scorer, USER)
    assert fast.shape == (len(rp.category_names()),)
    assert np.max(np.abs(fast - slow)) < 2e-3, np.max(np.abs(fast - slow))
    assert int(fast.argmax()) == int(slow.argmax())


def test_chunking_does_not_change_scores(scorer):
    full = scorer.scores(USER)
    scorer.chunk = 7
    try:
        chunked = scorer.scores(USER)
    finally:
        scorer.chunk = 55
    assert np.max(np.abs(full - chunked)) < 1e-4


def test_repeat_cache_on_legacy_tuples():
    from dxftrain.infer.torch_scorer import repeat_cache
    k = torch.arange(6.0).reshape(1, 1, 3, 2)
    out = repeat_cache(((k, k + 1),), 3)
    assert out[0][0].shape == (3, 1, 3, 2) and torch.equal(out[0][1][2], (k + 1)[0])


def test_make_scorer_picks_the_backend(monkeypatch):
    from dxftrain.infer import backend as bk
    monkeypatch.setenv("DXFTRAIN_BACKEND", "cpu")
    assert bk.resolve("auto") == "cpu"
    assert bk.base_model("cuda") == bk.train_config()["base_model_hf"]
    assert bk.base_model("mlx") == bk.train_config()["mlx"]["base_model"]
    assert bk.zero_shot_dir("data/datasets/ds-x", "mlx").name == "zeroshot-ds-x"       # existing MLX caches reused
    assert bk.zero_shot_dir("data/datasets/ds-x", "cuda").name == "zeroshot-ds-x-cuda"  # never mixed across backends
    with pytest.raises(SystemExit):
        bk.resolve("tpu")


def test_cuda_run_json_has_the_mlx_fields():
    from dxftrain.train.train_lora import epoch_history
    log = [{"loss": 2.1, "epoch": 0.5}, {"eval_loss": 0.40, "epoch": 1.0}, {"loss": 0.3, "epoch": 1.5},
           {"eval_loss": 0.35, "epoch": 2.0}, {"eval_loss": 0.37, "epoch": 3.0}]
    h = epoch_history(log, n_val=100)
    assert h["history"] == [{"epoch": 1, "val_loss": 0.4, "n_val": 100}, {"epoch": 2, "val_loss": 0.35, "n_val": 100},
                            {"epoch": 3, "val_loss": 0.37, "n_val": 100}]
    assert h["best_epoch"] == 2 and h["best_val_loss"] == 0.35 and len(h["train_log"]) == 2
