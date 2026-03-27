from __future__ import annotations

import numpy as np
from datasets import DatasetDict
from sklearn.model_selection import train_test_split


def make_train_val_test_splits(dataset_dict: DatasetDict, val_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Split the original train set into train + validation, keeping the original test set untouched.

    Args:
        dataset_dict: Hugging Face DatasetDict with at least 'train' and 'test'
        val_size: fraction of original train to use as validation
        seed: random seed

    Returns:
        DatasetDict with 'train', 'validation', 'test'
    """
    if "train" not in dataset_dict:
        raise ValueError("dataset_dict must contain a 'train' split.")
    if "test" not in dataset_dict:
        raise ValueError("dataset_dict must contain a 'test' split.")

    train_full = dataset_dict["train"]
    test_set = dataset_dict["test"]

    labels = np.array(train_full["label"])
    indices = np.arange(len(train_full))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    train_split = train_full.select(train_idx.tolist())
    val_split = train_full.select(val_idx.tolist())

    return DatasetDict(
        {
            "train": train_split,
            "validation": val_split,
            "test": test_set,
        }
    )


def print_split_info(dataset_dict: DatasetDict, split_name: str) -> None:
    split = dataset_dict[split_name]
    labels = np.array(split["label"])

    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n[INFO] Split: {split_name}")
    print(f"num_rows = {len(split)}")
    print("label_distribution =")
    for label, count in zip(unique, counts):
        pct = 100.0 * count / len(split)
        print(f"  label {label}: {count} samples ({pct:.2f}%)")