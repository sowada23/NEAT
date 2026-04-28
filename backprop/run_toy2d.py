#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DEFAULT_OUTPUTS = PROJECT_DIR / "outputs"


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".mplconfig"))
    sys.path.insert(0, str(SRC_DIR))
    from backprop_neat.cli.toy2d import main as cli_main

    argv = sys.argv[1:]
    if "--outputs" not in argv:
        argv = [*argv, "--outputs", str(DEFAULT_OUTPUTS)]
    cli_main(argv)


if __name__ == "__main__":
    main()
