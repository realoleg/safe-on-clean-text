import pandas as pd
from conftest import REPO

CHECKS = REPO / "results/checks"


def test_conformal_checks_passed():
    checks = pd.read_csv(CHECKS / "conformal_checks.csv")
    verdicts = checks[checks["key"].str.endswith("ok")]
    assert len(verdicts) > 0 and (verdicts["value"].astype(str) == "True").all()


def test_temperature_scaling_changed_no_set_in_any_run():
    checks = pd.read_csv(CHECKS / "temperature_invariance.csv")
    changed = checks[(checks["section"] == "t_invariance") & (checks["key"] == "set_rows_changed")]
    compared = checks[(checks["section"] == "t_invariance") & (checks["key"] == "set_rows_compared")]
    # Integrity of the committed check: one row per Dreaddit run with probabilities (TF-IDF + 15 encoder + 13 LLM runs).
    assert len(changed) == 29 and (changed["value"].astype(float) == 0).all()
    assert (compared["value"].astype(float) > 0).all()


def test_prevalence_bootstrap_check_passed():
    checks = pd.read_csv(CHECKS / "prevalence_bootstrap.csv")
    # Integrity of the committed check: 32 runs x 3 (method, scenario) x 2 alphas x 5 prevalences x 3 metrics.
    assert len(checks) == 2880 and checks["ok"].all()
