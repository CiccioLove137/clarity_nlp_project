from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
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
    text_column = get_cfg(config, "dataset.text_column", "text")
    label_column = get_cfg(config, "dataset.label_column", "label")

    print("\n[INFO] Loading dataset...")
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name)
    else:
        dataset = load_dataset(dataset_name)

    print("\n[INFO] Original dataset:")
    print(dataset)

    print("\n[INFO] Creating train/validation/test split...")
    dataset = make_train_val_test_splits(
    dataset,
    label_column=label_column,
    val_size=val_size,
    seed=seed,
    )

    print_split_info(dataset, "train", label_column)
    print_split_info(dataset, "validation", label_column)
    print_split_info(dataset, "test", label_column)

    print("\n[INFO] Building label mappings...")
    all_labels = sorted(set(dataset["train"][label_column]) | set(dataset["validation"][label_column]) | set(dataset["test"][label_column]))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: str(label) for label, idx in label2id.items()}

    print(f"[INFO] label2id: {label2id}")
    print(f"[INFO] id2label: {id2label}")

    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        texts = examples[text_column]
        labels = [label2id[label] for label in examples[label_column]]

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["label"] = labels
        return tokenized

    print("\n[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    print("\n[INFO] Tokenized dataset:")
    print(tokenized_dataset)

    num_labels = len(label2id)

    print("\n[INFO] Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={str(k): v for k, v in label2id.items()},
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