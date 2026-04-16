
"""
Manage output folders for training results.

This file helps keep saved results organized.
It creates a new numbered output folder so each training run
gets its own place for genomes, plots, and GIFs.

In short:
- checks existing output folders
- creates the next available output directory
- keeps results from different runs separate
"""

def build_output_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("output_"):
            suffix = path.name.removeprefix("output_")
            if suffix.isdigit():
                existing.append(int(suffix))

    next_idx = max(existing, default=0) + 1
    out_dir = base_dir / f"output_{next_idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir
