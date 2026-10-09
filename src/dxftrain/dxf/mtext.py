"""PLAN_INFO parsing: MTEXT formatting codes, \\P-delimited KEY=VALUE pairs."""

from __future__ import annotations

import re

from ezdxf.tools.text import plain_mtext

_KEY = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=\s*(.*?)\s*$")


def plain(raw: str) -> str:
    return plain_mtext(raw, split=False)


def parse_plan_info(raw: str) -> dict[str, str]:
    """KEY=VALUE pairs from a PLAN_INFO MTEXT (raw or already plain). Later duplicates do not overwrite."""
    out: dict[str, str] = {}
    for line in plain(raw).splitlines():
        m = _KEY.match(line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return out


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", value)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None
