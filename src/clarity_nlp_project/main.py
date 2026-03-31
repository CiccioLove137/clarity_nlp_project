from __future__ import annotations

import argparse
import random
import re
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.clarity_nlp_project.data.splits import (
    make_train_val_test_splits,
    print_split_info,
)
from src.clarity_nlp_project.training.trainer import train_model


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cfg(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current = config
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_3class_label(prediction_text: str) -> str | None:
    """
    Convert gpt3.5_prediction text into 3 final classes:
    - Clear Reply
    - Ambivalent
    - Clear Non-Reply
    """
    if prediction_text is None:
        return None

    text = str(prediction_text).lower()

    # 1.x => reply
    if "verdict: 1.1" in text or "explicit" in text:
        return "Clear Reply"

    if "verdict: 1.2" in text or "implicit" in text:
        return "Clear Reply"

    # 2.3 => ambivalent
    if "verdict: 2.3" in text or "partial/half-answer" in text or "partial answer" in text:
        return "Ambivalent"

    # 2.1 / 2.4 => clear non-reply
    if "verdict: 2.1" in text or "dodging" in text:
        return "Clear Non-Reply"

    if "verdict: 2.4" in text or "general" in text:
        return "Clear Non-Reply"

    # optional fallback for unexpected variants
    if "non-reply" in text:
        return "Clear Non-Reply"

    return None


def build_text(example: dict[str, Any]) -> str:
    question = str(example.get("interview_question", "")).strip()
    answer = str(example.get("interview_answer", "")).strip()

    return f"Question: {question}\nAnswer: {answer}"


def convert_split(raw_split) -> Dataset:
    texts = []
    labels = []

    for ex in raw_split:
        label = extract_3class_label(ex.get("gpt3.5_prediction"))
        text = build_text(ex)

        if label is None:
            continue

        if text.strip() == "":
            continue

        texts.append(text)
        labels.append(label)

    return Dataset.from_dict(
        {
            "text": texts,
            "label": labels,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(get_cfg(config, "training.seed", 42))
    set_seed(seed)

    dataset_name = get_cfg(config, "dataset.name", "ailsntua/QEvasion")
    dataset_config_name = get_cfg(config, "dataset.config_name", None)
    model_name = get_cfg(config, "model.name", "microsoft/deberta-v3-base")
    max_length = int(get_cfg(config, "dataset.max_length", 256))
    val_size = float(get_cfg(config, "dataset.val_size", 0.1))

    print("\n[INFO] Loading raw dataset...")
    if dataset_config_name:
        raw_dataset = load_dataset(dataset_name, dataset_config_name)
    else:
        raw_dataset = load_dataset(dataset_name)

    print("\n[INFO] Raw dataset:")
    print(raw_dataset)

    print("\n[INFO] Converting raw dataset into clean text/label format...")
    clean_dataset = DatasetDict(
        {
            "train": convert_split(raw_dataset["train"]),
            "test": convert_split(raw_dataset["test"]),
        }
    )

    print("\n[INFO] Clean dataset:")
    print(clean_dataset)

    print("\n[INFO] Creating train/validation/test split...")
    dataset = make_train_val_test_splits(
        clean_dataset,
        label_column="label",
        val_size=val_size,
        seed=seed,
    )

    print_split_info(dataset, "train", "label")
    print_split_info(dataset, "validation", "label")
    print_split_info(dataset, "test", "label")

    print("\n[INFO] Building label mappings...")
    unique_labels = sorted(set(dataset["train"]["label"]) | set(dataset["validation"]["label"]) | set(dataset["test"]["label"]))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"[INFO] label2id: {label2id}")
    print(f"[INFO] id2label: {id2label}")

    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["label"] = [label2id[label] for label in examples["label"]]
        return tokenized

    print("\n[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    )
    print("\n[DEBUG] Columns after tokenization:")
    print(tokenized_dataset["train"].column_names)
    print("\n[INFO] Tokenized dataset:")
    print(tokenized_dataset)

    num_labels = len(label2id)

    print("\n[INFO] Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("[INFO] Model loaded successfully!")

    print("\n[INFO] Starting training...")
    trainer = train_model(config, model, tokenized_dataset)

    print("\n[INFO] Running final evaluation on TEST set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])

    print("\n[INFO] Final TEST metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()