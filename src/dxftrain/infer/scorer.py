"""Label scoring: confidence from the model's likelihood of each label, never from its words.

For one object, every candidate answer '{"category": "<c>"}<|im_end|>' is scored by the sum of its token
log-probabilities given the prompt. Choosing the argmax over this closed set IS constrained decoding to the
label set. Calibrated confidence = softmax(scores / T).

MLX backend (Apple Silicon). Efficiency: the system prompt is identical for every object, so its KV cache is
computed once; per object only the user message is prefilled, then all labels are scored in one batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dxftrain.data import render_prompt as rp


def _kv(c):
    """(keys, values, offset) of an mlx-lm KVCache, trimmed to the filled length (buffers are padded)."""
    k, v, off = c.state
    return k[..., :off, :], v[..., :off, :], off


class MLXScorer:
    def __init__(self, model_path: str, adapter_path: str | None = None, labels: list[str] | None = None) -> None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.mx = mx
        self.make_cache = make_prompt_cache
        self.model, self.tok = load(model_path, adapter_path=adapter_path)
        self.labels = list(labels or rp.category_names())
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        cand = [self.tok.encode(rp.answer(c), add_special_tokens=False) + [self.end_id] for c in self.labels]
        self.cand_len = np.array([len(c) for c in cand])
        L = int(self.cand_len.max())
        pad = self.end_id
        self.cand = mx.array([c + [pad] * (L - len(c)) for c in cand])
        self.cand_np = np.array([c + [pad] * (L - len(c)) for c in cand])
        self.mask = mx.array((np.arange(L)[None, :] < self.cand_len[:, None]).astype(np.float32))
        # shared system prefix
        sys_ids = self._ids([{"role": "system", "content": rp.system_prompt()},
                             {"role": "user", "content": "x"}], gen=True)
        probe = self._ids([{"role": "system", "content": rp.system_prompt()},
                           {"role": "user", "content": "y"}], gen=True)
        k = 0
        while k < min(len(sys_ids), len(probe)) and sys_ids[k] == probe[k]:
            k += 1
        self.prefix = sys_ids[:k]
        self.prefix_cache = self.make_cache(self.model)
        self.model(mx.array([self.prefix]), cache=self.prefix_cache)
        mx.eval([c.state for c in self.prefix_cache])

    def _ids(self, msgs: list[dict], gen: bool) -> list[int]:
        return self.tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=True)

    def _cache_copy(self, batch: int):
        mx = self.mx
        new = self.make_cache(self.model)
        for src, dst in zip(self.prefix_cache, new):
            k, v, off = _kv(src)
            if batch > 1:
                k, v = mx.repeat(k, batch, axis=0), mx.repeat(v, batch, axis=0)
            else:
                k, v = k * 1, v * 1
            dst.state = (k, v, off)
        return new

    def scores(self, user: str) -> np.ndarray:
        """Summed log-probability of each label's answer tokens. Shape (n_labels,)."""
        mx = self.mx
        ids = self._ids([{"role": "system", "content": rp.system_prompt()}, {"role": "user", "content": user}], gen=True)
        assert ids[: len(self.prefix)] == self.prefix, "system prefix changed: prompt template drift"
        rest = ids[len(self.prefix):]
        cache = self._cache_copy(1)
        logits = self.model(mx.array([rest]), cache=cache)[0, -1]
        first = logits - mx.logsumexp(logits)                     # log p(first answer token)
        # tile the per-object cache to the label batch
        B = len(self.labels)
        for c in cache:
            k, v, off = _kv(c)
            c.state = (mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0), off)
        out = self.model(self.cand[:, :-1], cache=cache)          # predicts tokens 1..L-1
        lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
        tok = self.cand[:, 1:]
        gathered = mx.take_along_axis(lp, tok[..., None], axis=-1)[..., 0]
        s = first[self.cand[:, 0]] + (gathered * self.mask[:, 1:]).sum(axis=-1)
        mx.eval(s)
        return np.array(s, dtype=np.float64)


def score_rows(scorer: MLXScorer, rows: list[dict], cache_path: Path | None = None, log_every: int = 100) -> np.ndarray:
    """Scores for many rows, cached on disk by (object_id, variant, user-message hash) so reruns resume."""
    import hashlib
    import time
    done: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        for l in open(cache_path):
            d = json.loads(l)
            done[d["k"]] = d["s"]
    fh = open(cache_path, "a") if cache_path else None
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        user = r["messages"][1]["content"]
        k = r["object_id"] + "|" + r["variant"] + "|" + hashlib.sha256(user.encode()).hexdigest()[:16]
        if k not in done:
            s = scorer.scores(user).tolist()
            done[k] = s
            if fh:
                fh.write(json.dumps({"k": k, "s": s}) + "\n")
                fh.flush()
        out.append(done[k])
        if log_every and (i + 1) % log_every == 0:
            print(f"  scored {i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f}s/object", flush=True)
    if fh:
        fh.close()
    return np.array(out)
