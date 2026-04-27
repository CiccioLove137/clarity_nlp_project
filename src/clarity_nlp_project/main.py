from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.clarity_nlp_project.data.splits import (
    make_train_val_test_splits,
    print_split_info,
)
from src.clarity_nlp_project.training.trainer import train_model


SPECIAL_TOKENS = ["<QUESTION>", "</QUESTION>", "<ANSWER>", "</ANSWER>"]


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

        if not label:
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


def build_text(question: str, answer: str) -> str:
    return (
        "<QUESTION>\n"
        f"{question}\n"
        "</QUESTION>\n\n"
        "<ANSWER>\n"
        f"{answer}\n"
        "</ANSWER>"
    )


def build_global_attention_mask(
    input_ids: list[int],
    tokenizer,
    answer_global_tokens: int,
) -> list[int]:
    gam = [0] * len(input_ids)

    if len(gam) > 0:
        gam[0] = 1

    question_token_id = tokenizer.convert_tokens_to_ids("<QUESTION>")
    answer_token_id = tokenizer.convert_tokens_to_ids("<ANSWER>")

    for idx, token_id in enumerate(input_ids):
        if token_id == question_token_id:
            gam[idx] = 1

        if token_id == answer_token_id:
            gam[idx] = 1

            start = idx + 1
            end = min(idx + 1 + answer_global_tokens, len(gam))

            for j in range(start, end):
                gam[j] = 1

    return gam


def tokenize_split(
    split_dataset: Dataset,
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
    answer_global_tokens: int,
) -> Dataset:
    texts = []
    labels = []

    for question, answer, label in zip(
        split_dataset["question"],
        split_dataset["answer"],
        split_dataset["label"],
    ):
        question = str(question).strip() if question is not None else ""
        answer = str(answer).strip() if answer is not None else ""
        label = str(label).strip() if label is not None else ""

        if not question or not answer:
            continue

        if label not in label2id:
            continue

        text = build_text(question, answer)

        texts.append(text)
        labels.append(label2id[label])

    if len(texts) == 0:
        raise ValueError("This split became empty before tokenization.")

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    global_attention_mask = [
        build_global_attention_mask(
            input_ids=input_ids,
            tokenizer=tokenizer,
            answer_global_tokens=answer_global_tokens,
        )
        for input_ids in encoded["input_ids"]
    ]

    data = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "global_attention_mask": global_attention_mask,
        "labels": labels,
    }

    if "token_type_ids" in encoded:
        data["token_type_ids"] = encoded["token_type_ids"]

    return Dataset.from_dict(data)


def print_tokenizer_diagnostics(
    tokenizer,
    tokenized_dataset: DatasetDict,
    original_dataset: DatasetDict,
) -> None:
    print("\n" + "=" * 80)
    print("[CHECK] TOKENIZER DIAGNOSTICS")
    print("=" * 80)

    print("\n[CHECK] Special token IDs")
    for token in SPECIAL_TOKENS:
        print(f"{token} -> {tokenizer.convert_tokens_to_ids(token)}")

    for split_name in ["train", "validation", "test"]:
        lengths = [len(x) for x in tokenized_dataset[split_name]["input_ids"]]

        print(f"\n[CHECK] Token length statistics - {split_name}")
        print(f"num_examples = {len(lengths)}")
        print(f"min_len = {min(lengths)}")
        print(f"avg_len = {np.mean(lengths):.2f}")
        print(f"max_len = {max(lengths)}")
        print(f"num_at_max_length = {sum(1 for x in lengths if x == tokenizer.model_max_length)}")
        print(
            f"pct_at_max_length = "
            f"{100 * sum(1 for x in lengths if x == tokenizer.model_max_length) / len(lengths):.2f}%"
        )

    print("\n[CHECK] Global attention mask - train")
    for i in range(min(5, len(tokenized_dataset["train"]))):
        input_len = len(tokenized_dataset["train"][i]["input_ids"])
        gam = tokenized_dataset["train"][i]["global_attention_mask"]

        print(f"example_{i}:")
        print(f"  input_len = {input_len}")
        print(f"  gam_len = {len(gam)}")
        print(f"  gam_sum = {sum(gam)}")
        print(f"  first_40_gam = {gam[:40]}")

    print("\n[CHECK] Decoded tokenization example - train")

    example_index = 0
    input_ids = tokenized_dataset["train"][example_index]["input_ids"]
    label_id = tokenized_dataset["train"][example_index]["labels"]

    original_question = original_dataset["train"][example_index]["question"]
    original_answer = original_dataset["train"][example_index]["answer"]
    original_text = build_text(original_question, original_answer)

    print(f"example_index = {example_index}")
    print(f"label_id = {label_id}")
    print(f"tokenized_length = {len(input_ids)}")

    print("\n--- ORIGINAL TEXT ---")
    print(original_text)

    print("\n--- DECODED WITH SPECIAL TOKENS ---")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print("\n--- DECODED WITHOUT SPECIAL TOKENS ---")
    print(tokenizer.decode(input_ids, skip_special_tokens=True))

    print("\n" + "=" * 80)
    print("[CHECK] TOKENIZER DIAGNOSTICS COMPLETED")
    print("=" * 80)


def save_json(data: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = {}
    for key, value in data.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable[key] = float(value)
        else:
            serializable[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def evaluate_with_confusion_matrix(
    trainer,
    eval_dataset: Dataset,
    id2label: dict[int, str],
    output_dir: str,
    split_name: str = "test",
) -> None:
    print(f"\n[INFO] Computing predictions for {split_name.upper()} confusion matrix...")

    prediction_output = trainer.predict(eval_dataset)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    preds = np.argmax(logits, axis=-1)

    ordered_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in ordered_ids]

    cm = confusion_matrix(labels, preds, labels=ordered_ids)

    print(f"\n[INFO] Confusion Matrix - {split_name.upper()}:")
    print(cm)

    report = classification_report(
        labels,
        preds,
        labels=ordered_ids,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    print(f"\n[INFO] Classification Report - {split_name.upper()}:")
    print(report)

    os.makedirs(output_dir, exist_ok=True)

    cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrix.txt")
    report_path = os.path.join(output_dir, f"{split_name}_classification_report.txt")
    predictions_path = os.path.join(output_dir, f"{split_name}_predictions.json")

    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(str(cm))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    predictions_data = {
        "labels": labels.tolist(),
        "predictions": preds.tolist(),
        "label_names": label_names,
    }

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, indent=2)

    print(f"\n[INFO] Saved confusion matrix to: {cm_path}")
    print(f"[INFO] Saved classification report to: {report_path}")
    print(f"[INFO] Saved predictions to: {predictions_path}")


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
    output_dir = get_cfg(config, "training.output_dir", "outputs")
    answer_global_tokens = int(get_cfg(config, "dataset.answer_global_tokens", 20))

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

    added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )

    print("\n[INFO] Added special tokens to tokenizer:")
    print(f"special_tokens = {SPECIAL_TOKENS}")
    print(f"num_added_tokens = {added_tokens}")
    print(f"tokenizer_vocab_size = {len(tokenizer)}")

    print(f"\n[INFO] answer_global_tokens = {answer_global_tokens}")

    print("\n[INFO] Tokenizing train split...")
    tokenized_train = tokenize_split(
        dataset["train"],
        tokenizer,
        label2id,
        max_length,
        answer_global_tokens,
    )

    print("\n[INFO] Tokenizing validation split...")
    tokenized_validation = tokenize_split(
        dataset["validation"],
        tokenizer,
        label2id,
        max_length,
        answer_global_tokens,
    )

    print("\n[INFO] Tokenizing test split...")
    tokenized_test = tokenize_split(
        dataset["test"],
        tokenizer,
        label2id,
        max_length,
        answer_global_tokens,
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
    print("test:", set(tokenized_dataset["test"]["labels"]))

    print_tokenizer_diagnostics(
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        original_dataset=dataset,
    )

    num_labels = len(label2id)

    print("\n[INFO] Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("\n[INFO] Resizing model token embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    print(f"[INFO] New embedding size = {len(tokenizer)}")

    print("[INFO] Model loaded successfully!")

    print("\n[INFO] Starting training...")
    trainer = train_model(config, model, tokenizer, tokenized_dataset)

    print("\n[INFO] Running final evaluation on TEST set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])

    print("\n[INFO] Final TEST metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    evaluate_with_confusion_matrix(
        trainer=trainer,
        eval_dataset=tokenized_dataset["test"],
        id2label=id2label,
        output_dir=output_dir,
        split_name="test",
    )

    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()