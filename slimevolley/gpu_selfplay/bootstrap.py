from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SELFPLAY_DIR = THIS_FILE.parent
EXAMPLES_DIR = SELFPLAY_DIR.parent
NEAT_ROOT = EXAMPLES_DIR.parent

if str(NEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(NEAT_ROOT))

try:
    import jax  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slimevolley.gpu_selfplay requires the optional 'jax' dependency."
    ) from exc

from neat import NEATConfig, Population

__all__ = ["EXAMPLES_DIR", "NEATConfig", "Population"]
