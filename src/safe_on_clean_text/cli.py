"""
Command line of the study: `safe-on-clean-text <command>` (also `python -m safe_on_clean_text.cli <command>`).
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from safe_on_clean_text.config import load_config

DREADDIT = "configs/dreaddit.yaml"
SDCNL = "configs/sdcnl.yaml"


def prepare(config_paths: list[str]) -> None:
    from safe_on_clean_text.data import dreaddit, sdcnl
    from safe_on_clean_text.data.conditions import build_conditions
    from safe_on_clean_text.data.manifest import write_manifest

    for path in config_paths:
        config = load_config(path)
        # Each dataset has its own source and split procedure (data/dreaddit.py, data/sdcnl.py).
        (sdcnl if config["data"]["name"] == "sdcnl" else dreaddit).prepare(config)
        build_conditions(config)
        write_manifest(config)


def evaluate(config_paths: list[str]) -> None:
    from safe_on_clean_text.analyses.domain_shift import leave_one_domain_out
    from safe_on_clean_text.analyses.exploratory import addendum
    from safe_on_clean_text.analyses.mechanisms import mechanisms
    from safe_on_clean_text.analyses.official_test import official_test_p_values
    from safe_on_clean_text.analyses.prevalence import prevalence_shift
    from safe_on_clean_text.evaluation.checks import run_checks
    from safe_on_clean_text.evaluation.classification import classification_table
    from safe_on_clean_text.evaluation.pipeline import evaluate as evaluate_runs
    from safe_on_clean_text.models.llm.diagnostics import diagnostics

    configs = {path: load_config(path) for path in config_paths}
    for config in configs.values():
        evaluate_runs(config)
        if config["data"]["name"] == "dreaddit":
            # Analyses of the main dataset; SDCNL replicates the main evaluation only.
            run_checks(config)
            classification_table(config)
            diagnostics(config)
            official_test_p_values(config)
            prevalence_shift(config)
            leave_one_domain_out(config)
            mechanisms(config)
    if set(configs) == {DREADDIT, SDCNL}:
        addendum(configs[DREADDIT], configs[SDCNL])


def paper() -> None:
    from safe_on_clean_text.paper.build import build

    build(load_config(DREADDIT), load_config(SDCNL))


def report(args: argparse.Namespace) -> None:
    from safe_on_clean_text.reliability import reliability_report

    table = reliability_report(args.predictions, args.output_dir, args.config, args.methods, args.n_replicates)
    main_alpha = 0.1 if 0.1 in set(table["alpha"]) else table["alpha"].min()
    view = table[table["alpha"] == main_alpha].pivot_table(
        index=["condition", "method"], columns="metric", values="value", sort=False
    )
    print(
        f"alpha = {main_alpha}; medians over resplits (c1_below_band = share of resplits below the c1 reference band):"
    )
    print(view.round(4).to_string())


def train(args: argparse.Namespace) -> None:
    from safe_on_clean_text.models import baselines, encoders

    if args.target == "baselines":
        baselines.train(load_config(args.config))
    elif args.target == "encoders":
        encoders.train_all(args.config, args.models, args.seeds)
    elif args.target == "lodo":
        encoders.train_lodo(args.config)
    else:  # one unguarded run, used as the child process of `encoders` and `lodo`
        encoders.train_run(load_config(args.config), args.model, args.seed, args.lodo_domain)


def llm(args: argparse.Namespace) -> None:
    from safe_on_clean_text.models.llm import export, preflight, run

    if args.action == "run":
        run.run_model(args.config, args.model, args.worker)
    elif args.action == "export":
        export.export(load_config(args.config))
    else:
        preflight.preflight(load_config(args.config), args.models, args.methods)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="safe-on-clean-text", description=__doc__.strip())
    commands = root.add_subparsers(dest="command", required=True)

    def with_config(sub: argparse.ArgumentParser, default: str | None = DREADDIT, help_text: str = "") -> None:
        sub.add_argument("--config", default=default, help=help_text or f"Dataset config (default {default}).")

    def dataset_choice(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", help=f"Only this dataset config (default: {DREADDIT} and {SDCNL}).")

    dataset_choice(commands.add_parser("prepare", help="Download and split the datasets; build the degraded texts."))
    dataset_choice(commands.add_parser("evaluate", help="Evaluate every committed run and run all analyses."))
    commands.add_parser("paper", help="Build every paper table, figure and the results digest into results/paper/.")
    commands.add_parser("reproduce", help="prepare + evaluate + paper: the whole paper from the committed predictions.")

    report_parser = commands.add_parser("report", help="Reliability report of one predictions file.")
    report_parser.add_argument("predictions", help="Predictions CSV(.gz) in the run-folder format.")
    report_parser.add_argument(
        "--output-dir", required=True, help="Folder for long.parquet, summary.csv and report.csv."
    )
    with_config(report_parser)
    report_parser.add_argument(
        "--methods",
        nargs="+",
        choices=["M0", "M1", "M2", "M3", "M3-mix", "M5", "B1", "B2"],
        metavar="METHOD",
        help="Conformal methods and baselines: M0 M1 M2 M3 M3-mix M5 B1 B2 (default M0 M1 M3 M3-mix M5).",
    )
    report_parser.add_argument("--n-replicates", type=int, help="Grouped resplits (default from the config).")

    train_parser = commands.add_parser(
        "train", help="Train the TF-IDF baselines, the encoders or the leave-one-domain-out encoders."
    )
    # encoder-run (one unguarded run, the child process of `encoders` and `lodo`) is left out of the listing.
    targets = train_parser.add_subparsers(dest="target", required=True, metavar="{baselines,encoders,lodo}")
    with_config(targets.add_parser("baselines", help="TF-IDF + LogReg (C on validation) and LinearSVC."))
    encoders_parser = targets.add_parser(
        "encoders", help="Encoder runs under the Metal guard; complete runs are skipped."
    )
    with_config(encoders_parser)
    encoders_parser.add_argument(
        "--models",
        nargs="+",
        help="Encoder keys, e.g. distilbert roberta mentalroberta (default: all of encoders.models).",
    )
    encoders_parser.add_argument("--seeds", nargs="+", type=int, help="Seeds (default: encoders.seeds).")
    with_config(targets.add_parser("lodo", help="Leave-one-domain-out encoder runs (config `lodo`)."))
    run_parser = targets.add_parser("encoder-run")
    with_config(run_parser)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--lodo-domain")

    llm_parser = commands.add_parser(
        "llm", help="Zero-shot LLMs: answer from the cache or the model, export, preflight."
    )
    actions = llm_parser.add_subparsers(dest="action", required=True)
    llm_run = actions.add_parser("run", help="Answer the run plan of one LLM (resumable; cached answers make no call).")
    with_config(llm_run)
    llm_run.add_argument(
        "--model", required=True, choices=["gpt", "qwen", "gemma"], help="LLM to run (one of the config's llm.models)."
    )
    llm_run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    with_config(actions.add_parser("export", help="Cached answers -> run folders (no model calls)."))
    llm_preflight = actions.add_parser("preflight", help="Answer mass and parsing check on validation examples.")
    with_config(llm_preflight)
    llm_preflight.add_argument(
        "--models",
        nargs="+",
        default=["qwen", "gemma", "gpt"],
        choices=["gpt", "qwen", "gemma"],
        help="LLMs to check (default: qwen gemma gpt).",
    )
    llm_preflight.add_argument(
        "--methods", nargs="+", help="Only these confidence methods: s_tok s_verb s_cons (default: all)."
    )
    return root


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parser().parse_args(argv)
    datasets = (
        [args.config]
        if getattr(args, "config", None) and args.command in ("prepare", "evaluate")
        else [DREADDIT, SDCNL]
    )
    if args.command == "prepare":
        prepare(datasets)
    elif args.command == "evaluate":
        evaluate(datasets)
    elif args.command == "paper":
        paper()
    elif args.command == "reproduce":
        prepare([DREADDIT, SDCNL])
        evaluate([DREADDIT, SDCNL])
        paper()
    elif args.command == "report":
        report(args)
    elif args.command == "train":
        train(args)
    else:
        llm(args)


if __name__ == "__main__":
    main()
