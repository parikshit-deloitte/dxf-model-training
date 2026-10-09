"""Label scoring with PyTorch/transformers (NVIDIA GPU, or CPU for tests): the same method as infer/scorer.py.

For one object, every candidate answer '{"category": "<c>"}<|im_end|>' is scored by the sum of its token
log-probabilities given the prompt; the argmax over this closed set is constrained decoding to the label set, and
calibrated confidence = softmax(scores / T) (Stage D). Confidence never comes from generated text.

Efficiency: the prompt (system + user) is run once; its KV cache is repeated to a batch of candidates (in chunks of
`chunk` labels, to bound GPU memory) and every candidate's tokens are scored in one forward pass. Candidates are
right-padded with <|im_end|>; padding comes after a candidate's own tokens, so with causal attention it never changes
them, and it is masked out of the sum.

On CUDA the base model is loaded as in training (configs/qlora_12gb.yaml cuda: 4-bit nf4, compute dtype float16 on a
T4); a LoRA adapter from train_lora.py (PEFT) is applied on top. On CPU: float32, no quantisation (tests only).
"""

from __future__ import annotations

import copy

import numpy as np

from dxftrain.data import render_prompt as rp


def repeat_cache(past, n: int):
    """A copy of a prompt's KV cache repeated n times along the batch axis (legacy tuples or Cache objects)."""
    if isinstance(past, tuple):
        return tuple(tuple(t.repeat_interleave(n, dim=0) for t in layer) for layer in past)
    c = copy.deepcopy(past)
    if hasattr(c, "batch_repeat_interleave"):
        c.batch_repeat_interleave(n)
        return c
    if hasattr(c, "layers"):  # transformers >= 4.56: DynamicCache.layers[i].keys / .values
        for layer in c.layers:
            layer.keys = layer.keys.repeat_interleave(n, dim=0)
            layer.values = layer.values.repeat_interleave(n, dim=0)
        return c
    if hasattr(c, "key_cache"):
        c.key_cache = [k.repeat_interleave(n, dim=0) for k in c.key_cache]
        c.value_cache = [v.repeat_interleave(n, dim=0) for v in c.value_cache]
        return c
    raise TypeError(f"cannot repeat a {type(past).__name__} cache")


class TorchScorer:
    def __init__(self, model_path: str, adapter_path: str | None = None, labels: list[str] | None = None,
                 device: str = "cuda", chunk: int = 28) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from dxftrain.infer.backend import dtype_kw, train_config

        self.torch, self.chunk = torch, chunk
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("backend cuda, but torch sees no CUDA GPU")
        self.device = torch.device("cuda:0" if device == "cuda" else "cpu")
        kw: dict = dtype_kw(torch.float32)
        if device == "cuda":
            c = train_config()["cuda"]
            dtype = getattr(torch, c["compute_dtype"])
            kw = {**dtype_kw(dtype), "device_map": {"": 0}}
            if c.get("load_in_4bit", True):
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type=c["bnb_4bit_quant_type"],
                    bnb_4bit_use_double_quant=c["bnb_4bit_use_double_quant"],
                    bnb_4bit_compute_dtype=getattr(torch, c["bnb_4bit_compute_dtype"]))
        self.tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device) if device != "cuda" else model
        self.model.eval()
        self.labels = list(labels or rp.category_names())
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        cand = [self.tok.encode(rp.answer(c), add_special_tokens=False) + [self.end_id] for c in self.labels]
        self.cand_len = np.array([len(c) for c in cand])
        L = int(self.cand_len.max())
        self.cand = torch.tensor([c + [self.end_id] * (L - len(c)) for c in cand], device=self.device)
        self.mask = torch.tensor((np.arange(L)[None, :] < self.cand_len[:, None]).astype(np.float32), device=self.device)

    def _ids(self, msgs: list[dict]) -> list[int]:
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)

    def scores(self, user: str) -> np.ndarray:
        """Summed log-probability of each label's answer tokens. Shape (n_labels,), float64."""
        torch = self.torch
        ids = self._ids([{"role": "system", "content": rp.system_prompt()}, {"role": "user", "content": user}])
        n = len(ids)
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)          # log p(first answer token)
            parts = []
            for s in range(0, len(self.labels), self.chunk):
                cand = self.cand[s:s + self.chunk]
                b, L = cand.shape
                past = repeat_cache(out.past_key_values, b)
                inp = cand[:, :-1]                                                 # predicts tokens 1..L-1
                o = self.model(input_ids=inp, past_key_values=past, use_cache=True,
                               attention_mask=torch.ones((b, n + L - 1), dtype=torch.long, device=self.device),
                               position_ids=torch.arange(n, n + L - 1, device=self.device).expand(b, -1))
                lp = torch.log_softmax(o.logits.float(), dim=-1)
                got = lp.gather(-1, cand[:, 1:, None]).squeeze(-1)
                parts.append(first[cand[:, 0]] + (got * self.mask[s:s + b, 1:]).sum(-1))
            return torch.cat(parts).double().cpu().numpy()
