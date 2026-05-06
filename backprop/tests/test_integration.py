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
            "--hidden-activation",
            "relu",
            "--outputs",
            str(tmp_path),
        ]
    )
    out_dir = tmp_path / "output_1"
    assert (out_dir / "best_genome.pkl").exists()
    assert (out_dir / "accuracy.svg").exists()
    assert (out_dir / "fitness.svg").exists()
    assert (out_dir / "loss.svg").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "history.npz").exists()
    assert (out_dir / "summary.txt").exists()
    assert (out_dir / "decision_boundary.png").exists()
    assert (out_dir / "decision_boundary.gif").exists()
    assert (out_dir / "topology.png").exists()
    assert (out_dir / "topology.gif").exists()
    assert (out_dir / "decision_boundary" / "decision_boundary_generation_1.png").exists()
    assert (out_dir / "decision_boundary" / "decision_boundary_generation_2.png").exists()
    assert (out_dir / "topology" / "topology_generation_1.png").exists()
    assert (out_dir / "topology" / "topology_generation_2.png").exists()
    assert (out_dir / "decision_boundary.gif").read_bytes().startswith(b"GIF")
    assert (out_dir / "topology.gif").read_bytes().startswith(b"GIF")
    assert "hidden_activation: relu" in (out_dir / "summary.txt").read_text(encoding="utf-8")
    with open(out_dir / "best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    assert genome.forward([0.0, 1.0])
