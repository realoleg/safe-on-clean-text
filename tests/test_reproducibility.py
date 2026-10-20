import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import REPO, requires_prepared_data

from safe_on_clean_text.config import load_config
from safe_on_clean_text.evaluation.checks import run_checks


def same(produced: Path, committed: str) -> bool:
    return produced.read_bytes() == (REPO / committed).read_bytes()


def write_config(config: dict, path: Path) -> str:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


@pytest.mark.network
def test_data_preparation_reproduces_the_committed_outputs(tmp_path):
    dreaddit = copy.deepcopy(load_config(str(REPO / "configs/dreaddit.yaml")))
    dreaddit["data"]["processed_dir"] = str(tmp_path / "processed" / "dreaddit")
    sdcnl = copy.deepcopy(load_config(str(REPO / "configs/sdcnl.yaml")))
    sdcnl["data"]["raw_path"] = str(tmp_path / "raw" / "sdcnl.csv")
    sdcnl["data"]["processed_dir"] = str(tmp_path / "processed" / "sdcnl")
    for config in (dreaddit, sdcnl):
        config["paths"]["manifests_dir"] = str(tmp_path / "manifests")
    for config, name in [(dreaddit, "dreaddit.yaml"), (sdcnl, "sdcnl.yaml")]:
        command = [
            sys.executable,
            "-m",
            "safe_on_clean_text.cli",
            "prepare",
            "--config",
            write_config(config, tmp_path / name),
        ]
        subprocess.run(command, check=True, cwd=REPO)

    for dataset in ["dreaddit", "sdcnl"]:
        produced = tmp_path / "processed" / dataset
        for name in ["split_report.csv", "mix_assignment.csv", "conditions_report.csv"]:
            assert same(produced / name, f"data/processed/{dataset}/{name}"), (dataset, name)
    for name in ["dreaddit.csv", "sdcnl.csv", "checksums.sha256"]:
        assert same(tmp_path / "manifests" / name, f"data/manifests/{name}"), name


@requires_prepared_data
def test_conformal_checks_reproduce_the_committed_file(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    config = copy.deepcopy(load_config(str(REPO / "configs/dreaddit.yaml")))
    config["paths"]["checks_dir"] = str(tmp_path)
    run_checks(config)
    assert same(tmp_path / "conformal_checks.csv", "results/checks/conformal_checks.csv")
