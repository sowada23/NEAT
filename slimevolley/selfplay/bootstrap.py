from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SELFPLAY_DIR = THIS_FILE.parent
EXAMPLES_DIR = SELFPLAY_DIR.parent
NEAT_ROOT = EXAMPLES_DIR.parent
SLIME_ROOT = os.environ.get("SLIMEVOLLEYGYM_ROOT")

extra_paths = [NEAT_ROOT]
if SLIME_ROOT:
    extra_paths.append(Path(SLIME_ROOT).expanduser().resolve())

for p in extra_paths:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    import gym
except ImportError:
    import gymnasium as gym

from neat import NEATConfig, Population
import slimevolleygym  # noqa: F401
