from __future__ import annotations

from pathlib import Path


XOR_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = XOR_DIR / "output"


def build_output_dir(base_dir: Path | None = None) -> Path:
    root = OUTPUT_ROOT if base_dir is None else base_dir
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("output_"):
            suffix = path.name.removeprefix("output_")
            if suffix.isdigit():
                existing.append(int(suffix))

    next_idx = max(existing, default=0) + 1
    out_dir = root / f"output_{next_idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def latest_output_dir(base_dir: Path | None = None) -> Path:
    root = OUTPUT_ROOT if base_dir is None else base_dir
    if not root.exists():
        raise FileNotFoundError(f"No output directory found at {root}")

    candidates = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("output_"):
            suffix = path.name.removeprefix("output_")
            if suffix.isdigit():
                candidates.append((int(suffix), path))

    if not candidates:
        raise FileNotFoundError(f"No numbered output folders found in {root}")

    return max(candidates, key=lambda item: item[0])[1]
