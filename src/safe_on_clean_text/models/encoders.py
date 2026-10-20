from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import safetensors.torch
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from safe_on_clean_text.config import load_config
from safe_on_clean_text.data.splits import load_splits
from safe_on_clean_text.degradations import iter_prediction_inputs, load_condition_texts
from safe_on_clean_text.models.metal_guard import run_folder_complete, run_guarded, swap_usage
from safe_on_clean_text.models.predictions import (
    prediction_rows,
    check_probabilities,
    classification_metrics,
    predicted_labels,
    save_predictions,
    softmax,
)
from safe_on_clean_text.provenance import commit_hash, file_sha256, library_versions, timestamp, save_json

# A run is flagged as degenerate below this validation macro-F1 (flagged, never rerun or excluded).
DEGENERATE_VALIDATION_MACRO_F1 = 0.5
# save_pretrained writes LayerNorm tensors under these legacy names.
LEGACY_LAYER_NORM_NAMES = {".LayerNorm.gamma": ".LayerNorm.weight", ".LayerNorm.beta": ".LayerNorm.bias"}


def label_mappings() -> tuple[dict[str, int], dict[int, str]]:
    """
    Dreaddit is a binary stress classification task.
    """

    label2id = {
        "not_stress": 0,
        "stress": 1,
    }
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    desc: str,
) -> Dataset:
    """
    Convert a DataFrame into a tokenized Hugging Face dataset.
    """

    dataset = Dataset.from_pandas(df, preserve_index=False)

    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")

    def tokenize_batch(batch: dict[str, list]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True, desc=desc)


def build_tokenized_splits(
    split_to_df: dict[str, pd.DataFrame],
    model_name: str,
    max_length: int,
) -> tuple[Any, dict[str, Dataset]]:
    """
    Builds a tokenizer and tokenize each split into a Hugging Face dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_splits = {
        split_name: tokenize_dataframe(
            df=df,
            tokenizer=tokenizer,
            max_length=max_length,
            desc=f"Tokenizing {split_name}",
        )
        for split_name, df in split_to_df.items()
    }

    return tokenizer, tokenized_splits


def compute_trainer_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Metric function used by Hugging Face trainer.
    """

    logits, labels = eval_pred
    pred_labels = np.argmax(logits, axis=-1)
    return classification_metrics(labels, pred_labels)


def build_trainer(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    torch_empty_cache_steps: int,
    pad_to_multiple_of: int,
    seed: int,
    callbacks: list[TrainerCallback] | None = None,
) -> Trainer:
    """
    Create the model, tokenizer-related collation, training arguments, and trainer.
    """

    label2id, id2label = label_mappings()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
    )

    # Used for training, evaluation and every trainer.predict call (bounded set of batch shapes on MPS).
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        # transformers 5: a float < 1 is a fraction of total steps (warmup_ratio is deprecated).
        warmup_steps=warmup_ratio,
        # Releases the MPS allocator cache (safeguard; the per-shape graph caches are bounded by pad_to_multiple_of).
        torch_empty_cache_steps=torch_empty_cache_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_trainer_metrics,
        callbacks=callbacks,
    )

    return trainer


class EpochTimer(TrainerCallback):
    """
    Wall-clock seconds of each training epoch (excludes the end-of-epoch evaluation and saving).
    """

    def __init__(self) -> None:
        self.epoch_seconds: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_seconds.append(round(time.perf_counter() - self._start, 1))


def training_device() -> str:
    """
    Encoders train on MPS; without MPS they fall back to CPU with a warning.
    """

    if torch.backends.mps.is_available():
        return "mps"

    print("WARNING: MPS is not available, training on CPU.")
    return "cpu"


def predict_logits(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    texts: pd.Series,
    max_length: int,
    desc: str,
) -> np.ndarray:
    """
    Logits of shape (n_texts, 2), in the order of texts.
    Inputs are batched in order of token length (fewer batch shapes, less padding),
    then restored to the original order; the MPS cache is released after the pass.
    """

    dataset = tokenize_dataframe(
        df=pd.DataFrame({"text": texts.astype(str).to_numpy()}),
        tokenizer=tokenizer,
        max_length=max_length,
        desc=desc,
    )
    lengths = np.array([len(input_ids) for input_ids in dataset["input_ids"]])
    order = np.argsort(lengths, kind="stable")

    sorted_logits = np.asarray(trainer.predict(dataset.select(order)).predictions)
    logits = np.empty_like(sorted_logits)
    logits[order] = sorted_logits
    clear_device_cache()

    assert logits.shape == (len(texts), 2), logits.shape
    return logits


def checkpoint_tensors(checkpoint_dir: str | Path) -> tuple[dict[str, torch.Tensor], int]:
    """
    Tensors of a saved checkpoint file, legacy LayerNorm names (gamma / beta) mapped back to weight / bias.
    Also returns the number of renamed tensors.
    """

    tensors = safetensors.torch.load_file(Path(checkpoint_dir) / "model.safetensors", device="cpu")
    out: dict[str, torch.Tensor] = {}
    n_renamed = 0
    for name, tensor in tensors.items():
        for legacy, current in LEGACY_LAYER_NORM_NAMES.items():
            if name.endswith(legacy):
                name = name[: -len(legacy)] + current
                n_renamed += 1
        out[name] = tensor
    return out, n_renamed


def count_different_tensors(state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> int:
    """
    Number of shared tensor names whose values differ (compared on CPU, exact).
    """

    return sum(not torch.equal(state[name].detach().cpu(), reference[name]) for name in reference.keys() & state.keys())


def reload_best_checkpoint(trainer: Trainer) -> dict[str, Any]:
    """
    load_best_model_at_end restores the best checkpoint with load_state_dict(strict=False) and no key conversion,
    while save_pretrained writes every *.LayerNorm tensor under the legacy names gamma / beta: those tensors would keep
    their last-epoch values (2 tensors in DistilBERT, 50 in RoBERTa). Reload the best checkpoint through from_pretrained
    (with key conversion), then check the final model against the checkpoint file tensor-for-tensor.
    """

    checkpoint = trainer.state.best_model_checkpoint
    reference, n_renamed = checkpoint_tensors(checkpoint)
    model = trainer.model
    different_before = count_different_tensors(model.state_dict(), reference)

    best, loading_info = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_loading_info=True)
    model.load_state_dict(best.state_dict(), strict=True)
    del best

    final = model.state_dict()
    check = {
        "best_checkpoint": Path(checkpoint).name,
        "legacy_layer_norm_tensors_in_file": n_renamed,
        "tensors_different_before_reload": different_before,
        "loading_missing_keys": len(loading_info["missing_keys"]),
        "loading_unexpected_keys": len(loading_info["unexpected_keys"]),
        "loading_mismatched_keys": len(loading_info["mismatched_keys"]),
        "missing_keys": len(reference.keys() - final.keys()),
        "unexpected_keys": len(final.keys() - reference.keys()),
        "tensors_compared": len(reference),
        "tensors_different": count_different_tensors(final, reference),
    }
    check["passed"] = all(
        check[key] == 0
        for key in [
            "loading_missing_keys",
            "loading_unexpected_keys",
            "loading_mismatched_keys",
            "missing_keys",
            "unexpected_keys",
            "tensors_different",
        ]
    )
    assert check["passed"], check
    return check


class PeakMemory:
    """
    Peak MPS driver memory (GB), polled in a background thread between start() and stop(); None without MPS.
    """

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._enabled = torch.backends.mps.is_available()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, torch.mps.driver_allocated_memory())

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._enabled:
            self._thread.start()

    def stop(self) -> None:
        if self._enabled:
            self._stop.set()
            self._thread.join()
            self._sample()

    @property
    def peak_gb(self) -> float | None:
        return round(self.peak_bytes / 1024**3, 2) if self._enabled else None


def clear_device_cache() -> None:
    """
    Release cached MPS memory that no tensor uses (no effect on results).
    """

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def prepare_punkt(punkt_config: dict) -> None:
    """
    NLTK punkt_tab (the sentence splitter of the occlusion analysis) from its pinned URL into
    <nltk_dir>/tokenizers/punkt_tab, SHA-256 checked; NLTK then searches only nltk_dir, never ~/nltk_data.
    """

    nltk_dir = Path(punkt_config["nltk_dir"]).resolve()
    tokenizers = nltk_dir / "tokenizers"
    archive = tokenizers / "punkt_tab.zip"
    if not archive.exists():
        tokenizers.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl",
                "-sSL",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--max-time",
                "300",
                "-o",
                str(archive),
                punkt_config["url"],
            ],
            check=True,
        )
    digest = file_sha256(archive)
    if digest != punkt_config["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {archive}: {digest} != {punkt_config['sha256']}")
    if not (tokenizers / "punkt_tab").exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(tokenizers)
    nltk.data.path[:] = [str(nltk_dir)]


def split_sentences(text: str) -> list[str]:
    """
    Sentence split with NLTK punkt_tab (English); prepare_punkt must have run.
    """

    return nltk.sent_tokenize(text)


def sentence_occlusion(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    df: pd.DataFrame,
    max_length: int,
) -> pd.DataFrame:
    """
    For every sentence of every text with >= 2 sentences: change in p(stress) when the sentence is removed.
    The reference is the text re-joined from all its sentences, so the delta reflects only the removal.
    Position: first / last / middle sentence.
    """

    rows: list[dict[str, Any]] = []
    for example_id, label, text in zip(df["id"], df["label"], df["text"].astype(str)):
        sentences = split_sentences(text)
        n_sentences = len(sentences)
        if n_sentences < 2:
            continue

        rows.append(
            {
                "id": example_id,
                "label": label,
                "n_sentences": n_sentences,
                "sentence_idx": -1,
                "text": " ".join(sentences),
            }
        )
        for idx in range(n_sentences):
            rows.append(
                {
                    "id": example_id,
                    "label": label,
                    "n_sentences": n_sentences,
                    "sentence_idx": idx,
                    "text": " ".join(sentences[:idx] + sentences[idx + 1 :]),
                }
            )

    variants = pd.DataFrame(rows)
    logits = predict_logits(trainer, tokenizer, variants["text"], max_length, desc="Tokenizing occlusion")
    variants["prob_stress"] = softmax(logits)[:, 1]

    is_reference = variants["sentence_idx"] == -1
    reference = variants[is_reference].set_index("id")["prob_stress"]

    out = variants[~is_reference].drop(columns="text").reset_index(drop=True)
    out["prob_stress_full"] = out["id"].map(reference)
    out["delta_prob_stress"] = out["prob_stress"] - out["prob_stress_full"]
    out["position"] = np.select(
        [out["sentence_idx"] == 0, out["sentence_idx"] == out["n_sentences"] - 1],
        ["first", "last"],
        default="middle",
    )
    out = out.rename(columns={"prob_stress": "prob_stress_without"})

    return out[
        [
            "id",
            "label",
            "n_sentences",
            "sentence_idx",
            "position",
            "prob_stress_full",
            "prob_stress_without",
            "delta_prob_stress",
        ]
    ]


def seed_everything(seed: int) -> None:
    """
    Set random seed for transformer / numpy / pytorch if needed.
    """

    set_seed(seed)


def batch_settings(encoder_config: dict, model_key: str) -> tuple[int, int]:
    """
    (batch size, gradient accumulation steps) of a model: the per-model override, else the shared values.
    The effective batch must stay the shared one.
    """

    override = encoder_config.get("batch_overrides", {}).get(model_key, {})
    batch_size = int(override.get("batch_size", encoder_config["batch_size"]))
    accumulation_steps = int(override.get("gradient_accumulation_steps", encoder_config["gradient_accumulation_steps"]))
    shared = int(encoder_config["batch_size"]) * int(encoder_config["gradient_accumulation_steps"])
    assert batch_size * accumulation_steps == shared, (batch_size, accumulation_steps, shared)
    return batch_size, accumulation_steps


def train_run(config: dict, model_key: str, seed: int, lodo_domain: str | None = None) -> None:
    """
    One encoder run (no Metal guard; use train_all / train_lodo): fine-tune, reload and check the best checkpoint,
    predict every split and condition (and sentence occlusion), write the run folder and delete the checkpoint.
    """

    started_at = timestamp()
    start_time = time.perf_counter()

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    encoder_config = config["encoders"]
    conditions = config["conditions"]

    model_name = encoder_config["models"][model_key]
    max_length = int(encoder_config["max_length"])
    batch_size, accumulation_steps = batch_settings(encoder_config, model_key)

    run_tag = Path(dataset) if lodo_domain is None else Path(f"{dataset}_lodo") / lodo_domain
    if lodo_domain is not None:
        conditions = ["clean"]
    run_dir = Path(config["paths"]["runs_dir"]) / run_tag / model_key / f"seed_{seed}"
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"]) / run_tag / model_key / f"seed_{seed}"

    device = training_device()
    seed_everything(seed)

    # Checkpoints of an interrupted earlier attempt would mix into checkpoint rotation.
    if checkpoint_dir.exists():
        print("Removing leftover checkpoints: ", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)
    peak_memory = PeakMemory()
    peak_memory.start()

    split_to_df = load_splits(data_dir=data_dir)
    if lodo_domain is not None:
        assert lodo_domain in set(split_to_df["test"]["domain"]), lodo_domain
        split_to_df = {
            name: (df[df["domain"] == lodo_domain] if name == "test" else df[df["domain"] != lodo_domain]).reset_index(
                drop=True
            )
            for name, df in split_to_df.items()
        }
    condition_texts = load_condition_texts(data_dir / "conditions.csv")
    tokenizer, tokenized_splits = build_tokenized_splits(
        split_to_df={name: split_to_df[name][["text", "label"]] for name in ["train", "validation"]},
        model_name=model_name,
        max_length=max_length,
    )

    epoch_timer = EpochTimer()
    trainer = build_trainer(
        model_name=model_name,
        tokenizer=tokenizer,
        output_dir=checkpoint_dir,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=float(encoder_config["learning_rate"]),
        num_train_epochs=int(encoder_config["num_train_epochs"]),
        weight_decay=float(encoder_config["weight_decay"]),
        warmup_ratio=float(encoder_config["warmup_ratio"]),
        torch_empty_cache_steps=int(encoder_config["torch_empty_cache_steps"]),
        pad_to_multiple_of=int(encoder_config["pad_to_multiple_of"]),
        seed=seed,
        callbacks=[epoch_timer],
    )
    assert trainer.args.device.type == device, (trainer.args.device, device)

    train_start = time.perf_counter()
    trainer.train()
    train_seconds = round(time.perf_counter() - train_start, 1)
    best_checkpoint_check = reload_best_checkpoint(trainer)

    # Predictions with the best-epoch weights: validation clean only; calibration and test in every condition.
    prediction_frames: list[pd.DataFrame] = []
    clean_metrics: dict[str, dict[str, float]] = {}

    for split_name, condition, df, texts in iter_prediction_inputs(split_to_df, condition_texts, conditions):
        logits = predict_logits(trainer, tokenizer, texts, max_length, desc=f"Tokenizing {split_name}/{condition}")
        frame = prediction_rows(df["id"], split_name, condition, df["label"], texts, logits=logits)
        prediction_frames.append(frame)

        if condition == "clean":
            clean_metrics[split_name] = classification_metrics(frame["label"], predicted_labels(frame))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    check_probabilities(predictions[["prob_not_stress", "prob_stress"]].to_numpy(), "predictions")

    run_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, run_dir / "predictions.csv.gz")

    # Occlusion is a Dreaddit mechanism analysis (the SDCNL config also lists no occlusion models).
    with_occlusion = dataset == "dreaddit" and model_key in encoder_config["occlusion_models"] and lodo_domain is None
    if with_occlusion:
        prepare_punkt(encoder_config["punkt"])
        occlusion = sentence_occlusion(trainer, tokenizer, split_to_df["test"], max_length)
        check_probabilities(occlusion[["prob_stress_full", "prob_stress_without"]].to_numpy(), "occlusion")
        occlusion.to_csv(run_dir / "occlusion.csv", index=False)

    log_history = trainer.state.log_history
    best_metric = trainer.state.best_metric
    best_epoch = next(entry["epoch"] for entry in log_history if entry.get("eval_macro_f1") == best_metric)
    save_json(
        {
            "epoch_seconds": epoch_timer.epoch_seconds,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_metric,
            "log_history": log_history,
        },
        run_dir / "train_log.json",
    )

    # The run keeps only predictions: the checkpoint is not stored.
    shutil.rmtree(checkpoint_dir)
    peak_memory.stop()

    final_validation_f1 = clean_metrics["validation"]["macro_f1"]

    save_json(
        {
            "dataset": dataset,
            "model": model_key,
            "model_name": model_name,
            "seed": seed,
            "lodo_domain": lodo_domain,
            "commit": commit_hash(),
            "device": str(trainer.args.device),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "batch_mode": f"{batch_size}x{accumulation_steps}",
            # Batch 8 x 2 (or 4 x 4) replaces 16 x 1 after Metal out-of-memory errors (encoders.batch_overrides).
            "oom_fallback": accumulation_steps > 1,
            "pad_to_multiple_of": int(encoder_config["pad_to_multiple_of"]),
            "torch_empty_cache_steps": int(encoder_config["torch_empty_cache_steps"]),
            "conditions": conditions,
            "occlusion": with_occlusion,
            "n_rows": {name: len(df) for name, df in split_to_df.items()},
            "clean_metrics": clean_metrics,
            # Final model = best checkpoint, tensor-for-tensor (LayerNorm fix); its validation macro-F1 = the logged best.
            "best_epoch": best_epoch,
            "best_checkpoint_check": best_checkpoint_check,
            "validation_macro_f1_check": {
                "best_logged": best_metric,
                "final_model": final_validation_f1,
                "equal": final_validation_f1 == best_metric,
            },
            "degenerate": final_validation_f1 < DEGENERATE_VALIDATION_MACRO_F1,
            "degenerate_threshold_validation_macro_f1": DEGENERATE_VALIDATION_MACRO_F1,
            # MPS driver memory polled every 0.5 s over training, predictions and occlusion.
            "peak_mps_driver_gb": peak_memory.peak_gb,
            "train_steps": trainer.state.global_step,
            "train_seconds_per_step": round(train_seconds / trainer.state.global_step, 3),
            "train_seconds": train_seconds,
            "epoch_seconds": epoch_timer.epoch_seconds,
            "total_seconds": round(time.perf_counter() - start_time, 1),
            "started_at": started_at,
            "finished_at": timestamp(),
            "versions": library_versions(),
            "config": {
                "data": config["data"],
                "encoders": encoder_config,
                "conditions": conditions,
            },
        },
        run_dir / "run_meta.json",
    )

    print("Saved run folder to: ", run_dir.resolve())
    print(f"Best epoch: {best_epoch}, epoch seconds: {epoch_timer.epoch_seconds}, train seconds: {train_seconds}")
    print(f"Best-checkpoint check: {best_checkpoint_check}")
    print(
        f"Validation macro-F1 final / logged best: {final_validation_f1} / {best_metric}; "
        f"peak MPS driver memory: {peak_memory.peak_gb} GB"
    )
    print(pd.DataFrame(clean_metrics).T.round(4).to_string())


def train_all(config_path: str, models: list[str] | None = None, seeds: list[int] | None = None) -> None:
    """
    Encoder runs one at a time, each in a child process under the Metal guard; complete run folders are skipped.
    """

    config = load_config(config_path)
    runs_dir = Path(config["paths"]["runs_dir"]) / config["data"]["name"]
    for model in models or list(config["encoders"]["models"]):
        if model not in config["encoders"]["models"]:
            raise SystemExit(
                f"{model} is not an encoder of {config_path} (encoders.models: {list(config['encoders']['models'])})"
            )
        for seed in seeds or config["encoders"]["seeds"]:
            train_guarded(config_path, runs_dir / model / f"seed_{seed}", model, int(seed))


def train_lodo(config_path: str) -> None:
    """
    Leave-one-domain-out runs (config `lodo`): one per (held-out domain, seed), guarded like train_all.
    """

    config = load_config(config_path)
    if "lodo" not in config:
        raise SystemExit(f"{config_path} has no leave-one-domain-out section (lodo); it is a Dreaddit analysis")
    lodo = config["lodo"]
    root = Path(config["paths"]["runs_dir"]) / f"{config['data']['name']}_lodo"
    for domain in lodo["domains"]:
        for seed in lodo["seeds"]:
            train_guarded(config_path, root / domain / lodo["model"] / f"seed_{seed}", lodo["model"], int(seed), domain)


def train_guarded(config_path: str, run_dir: Path, model: str, seed: int, lodo_domain: str | None = None) -> None:
    """
    One encoder run in a child process. A run that prints a Metal command-buffer error is never complete: a
    run_meta.json written by the attempt is renamed to run_meta.failed.json and the loop stops.
    """

    if run_folder_complete(run_dir):
        print(f"[{timestamp()}] SKIP {run_dir}: run folder complete", flush=True)
        return
    print(f"[{timestamp()}] swap before: {swap_usage()}", flush=True)
    started = time.time()
    command = [
        sys.executable,
        "-m",
        "safe_on_clean_text.cli",
        "train",
        "encoder-run",
        "--config",
        config_path,
        "--model",
        model,
        "--seed",
        str(seed),
    ] + (["--lodo-domain", lodo_domain] if lodo_domain else [])
    n_errors = run_guarded(command)
    print(f"[{timestamp()}] swap after: {swap_usage()}", flush=True)
    if n_errors:
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists() and meta_path.stat().st_mtime >= started:
            meta_path.rename(run_dir / "run_meta.failed.json")
        raise RuntimeError(f"{run_dir}: {n_errors} Metal command-buffer error(s); run failed, training stopped.")
