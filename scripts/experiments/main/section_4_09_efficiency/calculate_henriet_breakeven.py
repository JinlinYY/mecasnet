"""Computational amortization for repeated cascade prediction and repair screening."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]




def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def break_even(preparation_seconds: float, simulator_seconds: float,
               surrogate_seconds: float) -> Optional[int]:
    savings = simulator_seconds - surrogate_seconds
    if savings <= 0:
        return None
    return int(math.ceil(preparation_seconds / savings))










