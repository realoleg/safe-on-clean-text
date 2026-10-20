import json

import numpy as np
import pandas as pd
import pytest
from conftest import REPO

RUN_FOLDERS = sorted(path.parent for path in (REPO / "results/runs").rglob("run_meta.json"))
REQUIRED = ["id", "split", "condition", "label", "n_words"]


def test_every_run_folder_is_found():
    # Dreaddit 30 (2 TF-IDF, 15 encoder, 13 LLM), leave-one-domain-out 15, SDCNL 8 (2 TF-IDF, 5 DistilBERT, 1 Qwen).
    assert len(RUN_FOLDERS) == 53


@pytest.mark.parametrize("run_dir", RUN_FOLDERS, ids=lambda p: str(p.relative_to(REPO / "results/runs")))
def test_run_folder(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")
    assert set(REQUIRED) <= set(predictions.columns)
    assert not any("text" in column for column in predictions.columns)
    assert not predictions.duplicated(["id", "split", "condition"]).any()
    assert len(meta["commit"]) == 40
    if "prob_stress" in predictions.columns:
        probabilities = predictions[["prob_not_stress", "prob_stress"]].to_numpy()
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    else:
        assert "decision_score" in predictions.columns  # LinearSVC
    if "best_checkpoint_check" in meta:  # encoder runs trained with the best-checkpoint reload check
        assert meta["best_checkpoint_check"]["passed"] and meta["validation_macro_f1_check"]["equal"]
    occlusion = run_dir / "occlusion.csv"
    if occlusion.exists():
        assert not any("text" in column or "sentence" == column for column in pd.read_csv(occlusion, nrows=1).columns)
