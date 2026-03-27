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
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def train_model(config: Any, model, tokenized_dataset):

    model_name = _get_attr(config, "model.name", "microsoft/deberta-v3-base")
    output_dir = _get_attr(config, "training.output_dir", "outputs")
    learning_rate = _get_attr(config, "training.learning_rate", 2e-5)
    train_batch_size = _get_attr(config, "training.per_device_train_batch_size", 8)
    eval_batch_size = _get_attr(config, "training.per_device_eval_batch_size", 8)
    num_train_epochs = _get_attr(config, "training.num_train_epochs", 3)

    os.makedirs(output_dir, exist_ok=True)

    # dataset
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

    # tokenizer + collator
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # training args (SUPER COMPATIBILI)
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy=eval_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        logging_strategy="epoch",
        report_to="none",
    )

    # trainer (API aggiornata)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
    )

    # train
    trainer.train()

    # eval
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print("\n[INFO] Evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    # save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer