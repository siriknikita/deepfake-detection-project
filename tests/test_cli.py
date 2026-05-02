"""CLI subcommand smoke tests.

Verifies that the parser builds, every subcommand has its expected
flags, and the bench / detect happy paths produce well-formed output.
The full train / eval workflows on real datasets are out of scope —
they are exercised by the dataset and training tests separately.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.cli import _build_parser, main


def test_parser_builds() -> None:
    parser = _build_parser()
    # Every documented subcommand is registered.
    sub = parser._subparsers._group_actions[0]
    assert {"detect", "train", "eval", "bench"} <= set(sub.choices.keys())


def test_help_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "Hyperplane-Forge" in captured.out


def test_detect_subcommand_runs_on_synthetic_image() -> None:
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        rgb = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
        img_path = Path(d) / "x.png"
        Image.fromarray(rgb).save(img_path)
        rc = main(
            [
                "detect",
                str(img_path),
                "--n-scales",
                "2",
                "--max-iter",
                "10",
            ],
        )
        assert rc == 0


def test_bench_subcommand_writes_csv() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rng = np.random.default_rng(0)
        for label in ("real", "fake"):
            (root / label).mkdir(parents=True, exist_ok=True)
            for i in range(2):
                arr = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
                Image.fromarray(arr).save(root / label / f"{i}.png")
        out = root / "feats.csv"
        rc = main(
            [
                "bench",
                "--dataset",
                "image-folder",
                "--data-root",
                str(root / "real"),
                "--fake-dir",
                str(root / "fake"),
                "--image-size",
                "32",
                "--out",
                str(out),
                "--max-iter",
                "10",
                "--n-scales",
                "2",
            ],
        )
        assert rc == 0
        assert out.exists()
        # Spot-check the CSV.
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 4
        assert "label" in df.columns
        assert "path" in df.columns


def test_eval_image_folder_requires_fake_dir() -> None:
    """Missing --fake-dir for --dataset=image-folder should raise SystemExit."""
    with tempfile.TemporaryDirectory() as d, pytest.raises(SystemExit):
        main(
            [
                "eval",
                "--dataset",
                "image-folder",
                "--data-root",
                str(Path(d)),
                "--image-size",
                "32",
            ],
        )
