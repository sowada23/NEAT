#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
python "${PROJECT_DIR}/run_toy2d.py" --dataset xor --generations 20 --population 40 --seed 7
