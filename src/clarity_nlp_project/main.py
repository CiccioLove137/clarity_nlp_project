from __future__ import annotations

import argparse
import random
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
    Map raw GPT verdict text to 3 final classes:
    - Clear Reply
    - Ambivalent
    - Clear Non-Reply
    """
    if prediction_text is None:
        return None

    text = str(prediction_text).lower()

    # Reply
    if "verdict: 1.1" in text or "explicit" in text:
        return "Clear Reply"

    if "verdict: 1.2" in text or "implicit" in text:
        return "Clear Reply"

    # Ambivalent
    if "verdict: 2.3" in text or "partial/half-answer" in text or "partial answer" in text:
        return "Ambivalent"

    # Clear Non-Reply
    if "verdict: 2.1" in text or "dodging" in text:
        return "Clear Non-Reply"

    if "verdict: 2.4" in text or "general" in text:
        return "Clear Non-Reply"

    # Fallback
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

        if not text.strip():
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

    print("\n[INFO] Checking empty texts...")
    empty_train = sum(1 for x in dataset["train"]["text"] if not str(x).strip())
    empty_val = sum(1 for x in dataset["validation"]["text"] if not str(x).strip())
    empty_test = sum(1 for x in dataset["test"]["text"] if not str(x).strip())
    print(f"empty_train: {empty_train}")
    print(f"empty_validation: {empty_val}")
    print(f"empty_test: {empty_test}")

    print("\n[INFO] Building label mappings...")
    unique_labels = sorted(
        set(dataset["train"]["label"])
        | set(dataset["validation"]["label"])
        | set(dataset["test"]["label"])
    )
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
        tokenized["labels"] = [label2id[label] for label in examples["label"]]
        return tokenized

    print("\n[INFO] Tokenizing dataset split by split...")

    tokenized_train = dataset["train"].map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing train",
    )

    tokenized_validation = dataset["validation"].map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing validation",
    )

    tokenized_test = dataset["test"].map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing test",
    )

    tokenized_dataset = DatasetDict(
        {
            "train": tokenized_train,
            "validation": tokenized_validation,
            "test": tokenized_test,
        }
    )

    print("\n[DEBUG] Tokenized train columns:")
    print(tokenized_dataset["train"].column_names)

    print("\n[DEBUG] Tokenized validation columns:")
    print(tokenized_dataset["validation"].column_names)

    print("\n[DEBUG] Tokenized test columns:")
    print(tokenized_dataset["test"].column_names)

    print("\n[DEBUG] Tokenized label ids:")
    print("train:", set(tokenized_dataset["train"]["labels"]))
    print("validation:", set(tokenized_dataset["validation"]["labels"]))

    if "labels" in tokenized_dataset["test"].column_names:
        print("test:", set(tokenized_dataset["test"]["labels"]))
    else:
        print("test: ⚠️ labels column missing")

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
    test_eval_dataset = tokenized_dataset["test"]

    allowed_columns = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    columns_to_remove = [c for c in test_eval_dataset.column_names if c not in allowed_columns]

    if columns_to_remove:
        test_eval_dataset = test_eval_dataset.remove_columns(columns_to_remove)

    print("[DEBUG] TEST columns used for final evaluation:")
    print(test_eval_dataset.column_names)

    test_metrics = trainer.evaluate(eval_dataset=test_eval_dataset)

    print("\n[INFO] Final TEST metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()