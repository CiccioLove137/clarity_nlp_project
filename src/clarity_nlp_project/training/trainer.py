from __future__ import annotations

import os
from typing import Any

import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _get_attr(config: Any, path: str, default: Any = None) -> Any:
    """
    Safely read nested attributes using dot notation.
    Example: _get_attr(config, "model.name", "microsoft/deberta-v3-base")
    """
    current = config
    for part in path.split("."):
        if current is None:
            return default

        if isinstance(current, dict):
            current = current.get(part, None)
        else:
            current = getattr(current, part, None)

    return default if current is None else current


def _compute_metrics(eval_pred):
    """
    Basic metrics without external dependencies.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = float((preds == labels).mean())

    return {
        "accuracy": accuracy,
    }


def train_model(config: Any, model, tokenized_dataset):
    """
    Train a Hugging Face classification model using Trainer.
    Assumes tokenized_dataset is a DatasetDict with at least 'train'
    and optionally 'test' / 'validation'.
    """

    model_name = _get_attr(config, "model.name", "microsoft/deberta-v3-base")
    output_dir = _get_attr(config, "training.output_dir", "outputs")
    learning_rate = _get_attr(config, "training.learning_rate", 2e-5)
    train_batch_size = _get_attr(config, "training.per_device_train_batch_size", 8)
    eval_batch_size = _get_attr(config, "training.per_device_eval_batch_size", 8)
    num_train_epochs = _get_attr(config, "training.num_train_epochs", 3)
    weight_decay = _get_attr(config, "training.weight_decay", 0.01)
    warmup_ratio = _get_attr(config, "training.warmup_ratio", 0.0)
    save_total_limit = _get_attr(config, "training.save_total_limit", 2)
    load_best_model_at_end = _get_attr(config, "training.load_best_model_at_end", True)
    metric_for_best_model = _get_attr(config, "training.metric_for_best_model", "accuracy")
    greater_is_better = _get_attr(config, "training.greater_is_better", True)
    fp16 = _get_attr(config, "training.fp16", False)
    report_to = _get_attr(config, "training.report_to", "none")
    seed = _get_attr(config, "training.seed", 42)

    os.makedirs(output_dir, exist_ok=True)

    if "train" not in tokenized_dataset:
        raise ValueError("tokenized_dataset must contain a 'train' split.")

    train_dataset = tokenized_dataset["train"]

    if "validation" in tokenized_dataset:
        eval_dataset = tokenized_dataset["validation"]
        eval_strategy = "epoch"
    elif "test" in tokenized_dataset:
        eval_dataset = tokenized_dataset["test"]
        eval_strategy = "epoch"
    else:
        eval_dataset = None
        eval_strategy = "no"
        load_best_model_at_end = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy=eval_strategy,
        save_strategy="epoch" if eval_dataset is not None else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=greater_is_better,
        fp16=fp16,
        report_to=report_to,
        seed=seed,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print("\n[INFO] Evaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer