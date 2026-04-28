from __future__ import annotations

import pickle

from backprop_neat.cli.toy2d import main


def test_tiny_cli_run_creates_artifacts(tmp_path):
    main(
        [
            "--dataset",
            "xor",
            "--generations",
            "1",
            "--population",
            "4",
            "--train-size",
            "24",
            "--test-size",
            "24",
            "--backprop-steps",
            "2",
            "--outputs",
            str(tmp_path),
        ]
    )
    out_dir = tmp_path / "output_1"
    assert (out_dir / "best_genome.pkl").exists()
    assert (out_dir / "history.npz").exists()
    assert (out_dir / "summary.txt").exists()
    assert (out_dir / "decision_boundary.png").exists()
    assert (out_dir / "topology.png").exists()
    with open(out_dir / "best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    assert genome.forward([0.0, 1.0])

