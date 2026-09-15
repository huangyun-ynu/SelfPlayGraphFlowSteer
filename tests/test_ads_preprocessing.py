from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from selfplay_graph_flowsteer.ads_preprocessing import (
    ADSPreprocessingConfig,
    _mean_token_nll,
    build_ads_records,
)


def test_build_ads_records_drops_offline_explicit_targets(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"webshop-{index}",
            "_ads_sample_id": f"webshop-{index}",
            "dataset": "webshop",
            "split": "train",
            "prompt": f"goal {index}",
            "ads_target": {"asin": f"private-{index}"},
            "metadata": {},
        }
        for index in range(3)
    ]
    records, _manifest, _features, _labels = build_ads_records(
        rows,
        np.eye(3, dtype=np.float32),
        np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        ["ads_target"] * 3,
        ADSPreprocessingConfig(
            model_path=tmp_path,
            output_path=tmp_path / "pool.jsonl",
            artifacts_dir=tmp_path / "artifacts",
            num_clusters=2,
            pca_dim=2,
        ),
    )

    assert all("ads_target" not in record for record in records)
    assert "private-" not in repr(records)


def test_chunked_mean_token_nll_matches_direct_cross_entropy() -> None:
    generator = torch.Generator().manual_seed(20260915)
    logits = torch.randn(17, 31, generator=generator, dtype=torch.bfloat16)
    targets = torch.randint(0, 31, (17,), generator=generator)

    expected = functional.cross_entropy(logits.float(), targets).item()

    assert _mean_token_nll(logits, targets, chunk_size=4) == pytest.approx(
        expected, rel=1e-6
    )
