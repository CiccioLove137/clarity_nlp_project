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


def convert_split(raw_split) -> Dataset:
    questions = []
    answers = []
    labels = []

    for ex in raw_split:
        question = str(ex.get("interview_question", "")).strip()
        answer = str(ex.get("interview_answer", "")).strip()
        label = ex.get("clarity_label")

        if label is None:
            continue

        label = str(label).strip()
        if label == "":
            continue

        if not question or not answer:
            continue

        questions.append(question)
        answers.append(answer)
        labels.append(label)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "label": labels,
        }
    )


def tokenize_split(split_dataset: Dataset, tokenizer, label2id: dict[str, int], max_length: int) -> Dataset:
    questions = []
    answers = []
    labels = []

    for question, answer, label in zip(
        split_dataset["question"],
        split_dataset["answer"],
        split_dataset["label"],
    ):
        question = str(question) if question is not None else ""
        answer = str(answer) if answer is not None else ""
        label = str(label).strip() if label is not None else ""

        if not question.strip() or not answer.strip():
            continue

        if label not in label2id:
            continue

        questions.append(question)
        answers.append(answer)
        labels.append(label2id[label])

    if len(questions) == 0:
        raise ValueError("This split became empty before tokenization.")

    # Tokenizzazione in modalità sequence pair: question + answer
    encoded = tokenizer(
        questions,
        answers,
        truncation="longest_first",
        padding=False,
        max_length=max_length,
    )

    # Longformer: mettiamo attenzione globale sul primo token (<s>)
    global_attention_mask = []
    for input_ids in encoded["input_ids"]:
        gam = [0] * len(input_ids)
        if len(gam) > 0:
            gam[0] = 1
        global_attention_mask.append(gam)

    data = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "global_attention_mask": global_attention_mask,
        "labels": labels,
    }

    if "token_type_ids" in encoded:
        data["token_type_ids"] = encoded["token_type_ids"]

    return Dataset.from_dict(data)


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
    model_name = get_cfg(config, "model.name", "allenai/longformer-base-4096")
    max_length = int(get_cfg(config, "dataset.max_length", 4096))
    val_size = float(get_cfg(config, "dataset.val_size", 0.1))

    print("\n[INFO] Loading raw dataset...")
    if dataset_config_name:
        raw_dataset = load_dataset(dataset_name, dataset_config_name)
    else:
        raw_dataset = load_dataset(dataset_name)

    print("\n[INFO] Raw dataset:")
    print(raw_dataset)

    print("\n[INFO] Converting raw dataset into clean question/answer/label format...")
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

    print("\n[INFO] Checking empty fields...")
    empty_train_q = sum(1 for x in dataset["train"]["question"] if not str(x).strip())
    empty_val_q = sum(1 for x in dataset["validation"]["question"] if not str(x).strip())
    empty_test_q = sum(1 for x in dataset["test"]["question"] if not str(x).strip())

    empty_train_a = sum(1 for x in dataset["train"]["answer"] if not str(x).strip())
    empty_val_a = sum(1 for x in dataset["validation"]["answer"] if not str(x).strip())
    empty_test_a = sum(1 for x in dataset["test"]["answer"] if not str(x).strip())

    print(f"empty_train_question: {empty_train_q}")
    print(f"empty_validation_question: {empty_val_q}")
    print(f"empty_test_question: {empty_test_q}")
    print(f"empty_train_answer: {empty_train_a}")
    print(f"empty_validation_answer: {empty_val_a}")
    print(f"empty_test_answer: {empty_test_a}")

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

    print("\n[INFO] Tokenizing train split...")
    tokenized_train = tokenize_split(dataset["train"], tokenizer, label2id, max_length)

    print("\n[INFO] Tokenizing validation split...")
    tokenized_validation = tokenize_split(dataset["validation"], tokenizer, label2id, max_length)

    print("\n[INFO] Tokenizing test split...")
    tokenized_test = tokenize_split(dataset["test"], tokenizer, label2id, max_length)

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
    print("test:", set(tokenized_dataset["test"]["labels"]))

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