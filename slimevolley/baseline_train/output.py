from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_output_dir(base_dir: str | Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    out_dir = base_dir / run_id
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_id}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=False, exist_ok=False)
    return out_dir

