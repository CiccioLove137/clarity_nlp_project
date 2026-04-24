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


SPECIAL_TOKENS = [
    "<QUESTION>",
    "</QUESTION>",
    "<ANSWER>",
    "</ANSWER>",
]


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


def add_special_tokens_to_tokenizer(tokenizer) -> int:
    """
    Aggiunge marker strutturali per distinguere chiaramente domanda e risposta.
    Restituisce il numero di token aggiunti.
    """
    special_tokens_dict = {
        "additional_special_tokens": SPECIAL_TOKENS
    }

    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print("\n[INFO] Added special tokens to tokenizer:")
    print(f"special_tokens = {SPECIAL_TOKENS}")
    print(f"num_added_tokens = {num_added_tokens}")
    print(f"tokenizer_vocab_size = {len(tokenizer)}")

    return num_added_tokens


def tokenize_split(
    split_dataset: Dataset,
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
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

        combined_text = (
            "<QUESTION>\n"
            f"{question}\n"
            "</QUESTION>\n\n"
            "<ANSWER>\n"
            f"{answer}\n\n"
            f"{answer}\n"
            "</ANSWER>"
        )

        texts.append(combined_text)
        labels.append(label2id[label])

    if len(texts) == 0:
        raise ValueError("This split became empty before tokenization.")

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    question_token_id = tokenizer.convert_tokens_to_ids("<QUESTION>")
    answer_token_id = tokenizer.convert_tokens_to_ids("<ANSWER>")

    global_attention_mask = []

    for input_ids in encoded["input_ids"]:
        gam = [0] * len(input_ids)

        if len(gam) > 0:
            gam[0] = 1

        for i, token_id in enumerate(input_ids):
            if token_id == question_token_id or token_id == answer_token_id:
                gam[i] = 1

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


def analyze_tokenized_split(split_name: str, tokenized_split: Dataset, max_length: int) -> None:
    lengths = [len(input_ids) for input_ids in tokenized_split["input_ids"]]

    num_examples = len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / num_examples
    num_at_max_length = sum(1 for length in lengths if length == max_length)
    pct_at_max_length = 100.0 * num_at_max_length / num_examples

    print(f"\n[CHECK] Token length statistics - {split_name}")
    print(f"num_examples = {num_examples}")
    print(f"min_len = {min_len}")
    print(f"avg_len = {avg_len:.2f}")
    print(f"max_len = {max_len}")
    print(f"num_at_max_length = {num_at_max_length}")
    print(f"pct_at_max_length = {pct_at_max_length:.2f}%")


def check_global_attention_mask(split_name: str, tokenized_split: Dataset, n_examples: int = 5) -> None:
    print(f"\n[CHECK] Global attention mask - {split_name}")

    n_examples = min(n_examples, len(tokenized_split))

    for i in range(n_examples):
        gam = tokenized_split[i]["global_attention_mask"]
        input_ids = tokenized_split[i]["input_ids"]

        print(f"example_{i}:")
        print(f"  input_len = {len(input_ids)}")
        print(f"  gam_len = {len(gam)}")
        print(f"  gam_sum = {sum(gam)}")
        print(f"  first_30_gam = {gam[:30]}")

        if len(gam) != len(input_ids):
            print("  [WARNING] global_attention_mask length is different from input_ids length.")

        if len(gam) > 0 and gam[0] != 1:
            print("  [WARNING] first token does not have global attention.")

        if sum(gam) < 1:
            print("  [WARNING] no global attention tokens found.")


def inspect_decoded_example(
    dataset_split: Dataset,
    tokenized_split: Dataset,
    tokenizer,
    split_name: str,
    example_index: int = 0,
    max_chars: int = 3000,
) -> None:
    if len(tokenized_split) == 0:
        print(f"\n[CHECK] Cannot decode example from {split_name}: split is empty.")
        return

    example_index = min(example_index, len(tokenized_split) - 1)

    original_text = (
        "<QUESTION>\n"
        f"{dataset_split[example_index]['question']}\n"
        "</QUESTION>\n\n"
        "<ANSWER>\n"
        f"{dataset_split[example_index]['answer']}\n\n"
        f"{dataset_split[example_index]['answer']}\n"
        "</ANSWER>"
    )

    decoded_with_special_tokens = tokenizer.decode(
        tokenized_split[example_index]["input_ids"],
        skip_special_tokens=False,
    )

    decoded_without_special_tokens = tokenizer.decode(
        tokenized_split[example_index]["input_ids"],
        skip_special_tokens=False,
    )

    print(f"\n[CHECK] Decoded tokenization example - {split_name}")
    print(f"example_index = {example_index}")
    print(f"label_id = {tokenized_split[example_index]['labels']}")
    print(f"tokenized_length = {len(tokenized_split[example_index]['input_ids'])}")

    print("\n--- ORIGINAL TEXT ---")
    print(original_text[:max_chars])

    print("\n--- DECODED WITH SPECIAL TOKENS ---")
    print(decoded_with_special_tokens[:max_chars])

    print("\n--- DECODED TEXT ---")
    print(decoded_without_special_tokens[:max_chars])


def check_special_token_ids(tokenizer) -> None:
    print("\n[CHECK] Special token IDs")

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token} -> {token_id}")

        if token_id == tokenizer.unk_token_id:
            print(f"[WARNING] {token} is mapped to unk_token_id.")


def run_tokenizer_checks(
    dataset: DatasetDict,
    tokenized_dataset: DatasetDict,
    tokenizer,
    max_length: int,
) -> None:
    print("\n" + "=" * 80)
    print("[CHECK] TOKENIZER DIAGNOSTICS")
    print("=" * 80)

    check_special_token_ids(tokenizer)

    analyze_tokenized_split("train", tokenized_dataset["train"], max_length)
    analyze_tokenized_split("validation", tokenized_dataset["validation"], max_length)
    analyze_tokenized_split("test", tokenized_dataset["test"], max_length)

    check_global_attention_mask("train", tokenized_dataset["train"], n_examples=5)

    inspect_decoded_example(
        dataset_split=dataset["train"],
        tokenized_split=tokenized_dataset["train"],
        tokenizer=tokenizer,
        split_name="train",
        example_index=0,
    )

    print("\n" + "=" * 80)
    print("[CHECK] TOKENIZER DIAGNOSTICS COMPLETED")
    print("=" * 80)


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
    num_added_tokens = add_special_tokens_to_tokenizer(tokenizer)

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

    run_tokenizer_checks(
        dataset=dataset,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    num_labels = len(label2id)

    print("\n[INFO] Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if num_added_tokens > 0:
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

    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()