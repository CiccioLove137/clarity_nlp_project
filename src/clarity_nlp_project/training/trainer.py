from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


class StableTrainer(Trainer):
    """
    Trainer più robusto numericamente.
    Se i logits contengono NaN/Inf, li sanitizza prima di calcolare la loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Stabilizzazione numerica
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not torch.isfinite(loss):
            loss = torch.zeros((), device=logits.device, requires_grad=True)

        if return_outputs:
            outputs["logits"] = logits
            return loss, outputs
        return loss


def train_model(config: Any, model, tokenized_dataset):
    model_name = _get_attr(config, "model.name", "microsoft/deberta-v3-base")
    output_dir = _get_attr(config, "training.output_dir", "outputs")

    learning_rate = _to_float(_get_attr(config, "training.learning_rate", 1e-5), 1e-5)
    train_batch_size = _to_int(_get_attr(config, "training.per_device_train_batch_size", 4), 4)
    eval_batch_size = _to_int(_get_attr(config, "training.per_device_eval_batch_size", 4), 4)
    num_train_epochs = _to_int(_get_attr(config, "training.num_train_epochs", 3), 3)
    fp16 = _to_bool(_get_attr(config, "training.fp16", False), False)
    weight_decay = _to_float(_get_attr(config, "training.weight_decay", 0.01), 0.01)
    warmup_ratio = _to_float(_get_attr(config, "training.warmup_ratio", 0.1), 0.1)
    gradient_accumulation_steps = _to_int(
        _get_attr(config, "training.gradient_accumulation_steps", 1), 1
    )

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("\n[INFO] Training configuration:")
    print(f"model_name = {model_name}")
    print(f"output_dir = {output_dir}")
    print(f"learning_rate = {learning_rate} ({type(learning_rate).__name__})")
    print(f"train_batch_size = {train_batch_size} ({type(train_batch_size).__name__})")
    print(f"eval_batch_size = {eval_batch_size} ({type(eval_batch_size).__name__})")
    print(f"num_train_epochs = {num_train_epochs} ({type(num_train_epochs).__name__})")
    print(f"fp16 = {fp16} ({type(fp16).__name__})")
    print(f"weight_decay = {weight_decay}")
    print(f"warmup_ratio = {warmup_ratio}")
    print(f"gradient_accumulation_steps = {gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy=eval_strategy,
        save_strategy="epoch" if eval_dataset is not None else "no",
        logging_strategy="epoch",
        report_to="none",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        max_grad_norm=1.0,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = StableTrainer(
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
        for k, v in metrics.items():
            print(f"{k}: {v}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer