"""One-line progress for long steps: plain lines (no \\r), flushed, so they read the same on a terminal, in nohup
logs and through `make watch`. Every line starts with a timestamp and has done/total, %, rate and ETA."""

from __future__ import annotations

import datetime as dt
import time


def hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def gpu_mem() -> str:
    """'GPU 9.8/15.6 GB' on CUDA, '' elsewhere (never imports torch itself)."""
    import sys
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return ""
    free, total = torch.cuda.mem_get_info()
    return f"GPU {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB"


class Progress:
    """p = Progress('score val', total); p.update(done, extra='...') prints at most every `every_s` seconds."""

    def __init__(self, label: str, total: int, every_s: float = 15.0):
        self.label, self.total, self.every_s = label, max(1, total), every_s
        self.t0 = time.time()
        self.last = 0.0

    def update(self, done: int, extra: str = "", force: bool = False, worked: int | None = None) -> None:
        """`worked`: items actually computed this run (default `done`); cached/resumed items do not set the rate."""
        now = time.time()
        if not force and now - self.last < self.every_s and done < self.total:
            return
        self.last = now
        worked = done if worked is None else worked
        rate = (now - self.t0) / worked if worked > 0 else 0.0
        eta = rate * (self.total - done)
        parts = [f"{dt.datetime.now():%H:%M:%S}", f"[{self.label}]", f"{done}/{self.total}",
                 f"{100 * done / self.total:5.1f}%", f"elapsed {hms(now - self.t0)}", f"ETA {hms(eta)}"]
        if rate:
            parts.append(f"{rate:.2f}s/it")
        if extra:
            parts.append(extra)
        print("  ".join(parts), flush=True)
