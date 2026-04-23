from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
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


def build_metrics_fn(id2label: dict[int, str]):
    ordered_ids = sorted(id2label.keys())

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, preds)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )

        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            labels=ordered_ids,
            zero_division=0,
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
        }

        for idx, class_id in enumerate(ordered_ids):
            class_name = id2label[class_id].replace(" ", "_")
            metrics[f"precision_{class_name}"] = float(precision_per_class[idx])
            metrics[f"recall_{class_name}"] = float(recall_per_class[idx])
            metrics[f"f1_{class_name}"] = float(f1_per_class[idx])
            metrics[f"support_{class_name}"] = float(support_per_class[idx])

        return metrics

    return _compute_metrics


def save_metrics_to_json(metrics: dict, output_path: str) -> None:
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)


def build_sample_weights(labels: list[int], num_classes: int) -> torch.DoubleTensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[label] for label in labels]
    return torch.DoubleTensor(sample_weights)


def build_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)

    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class BalancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        use_weighted_sampler: bool = True,
        train_labels: list[int] | None = None,
        num_classes: int | None = None,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_weighted_sampler = use_weighted_sampler
        self.train_labels = train_labels
        self.num_classes = num_classes
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not torch.isfinite(loss):
            loss = torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)

        if return_outputs:
            return loss, outputs
        return loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if not self.use_weighted_sampler:
            return super().get_train_dataloader()

        if self.train_labels is None or self.num_classes is None:
            raise ValueError("Weighted sampler requested but train_labels/num_classes are missing.")

        sample_weights = build_sample_weights(self.train_labels, self.num_classes)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def train_model(config: Any, model, tokenizer, tokenized_dataset):
    output_dir = _get_attr(config, "training.output_dir", "outputs")

    learning_rate = _to_float(_get_attr(config, "training.learning_rate", 2e-5), 2e-5)
    train_batch_size = _to_int(_get_attr(config, "training.per_device_train_batch_size", 1), 1)
    eval_batch_size = _to_int(_get_attr(config, "training.per_device_eval_batch_size", 1), 1)
    num_train_epochs = _to_int(_get_attr(config, "training.num_train_epochs", 5), 5)
    fp16 = _to_bool(_get_attr(config, "training.fp16", False), False)
    weight_decay = _to_float(_get_attr(config, "training.weight_decay", 0.01), 0.01)
    warmup_steps = _to_int(_get_attr(config, "training.warmup_steps", 100), 100)
    gradient_accumulation_steps = _to_int(
        _get_attr(config, "training.gradient_accumulation_steps", 8), 8
    )
    use_weighted_sampler = _to_bool(_get_attr(config, "training.use_weighted_sampler", True), True)
    load_best_model_at_end = _to_bool(
        _get_attr(config, "training.load_best_model_at_end", True), True
    )
    early_stopping_patience = _to_int(
        _get_attr(config, "training.early_stopping_patience", 2), 2
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

    train_labels = list(train_dataset["labels"])
    num_classes = model.config.num_labels
    id2label = model.config.id2label

    class_weights = build_class_weights(train_labels, num_classes)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if use_weighted_sampler:
        counts = np.bincount(train_labels, minlength=num_classes)
        print("\n[INFO] WeightedRandomSampler enabled:")
        print(f"class_counts = {counts.tolist()}")

    print("\n[INFO] Training configuration:")
    print(f"output_dir = {output_dir}")
    print(f"learning_rate = {learning_rate}")
    print(f"train_batch_size = {train_batch_size}")
    print(f"eval_batch_size = {eval_batch_size}")
    print(f"num_train_epochs = {num_train_epochs}")
    print(f"fp16 = {fp16}")
    print(f"weight_decay = {weight_decay}")
    print(f"warmup_steps = {warmup_steps}")
    print(f"gradient_accumulation_steps = {gradient_accumulation_steps}")
    print(f"use_weighted_sampler = {use_weighted_sampler}")
    print(f"load_best_model_at_end = {load_best_model_at_end}")
    print(f"class_weights = {class_weights.tolist()}")

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
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        metric_for_best_model="f1_macro" if eval_dataset is not None else None,
        greater_is_better=True,
        load_best_model_at_end=load_best_model_at_end if eval_dataset is not None else False,
        save_total_limit=2,
    )

    callbacks = []
    if eval_dataset is not None and load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_metrics_fn(id2label) if eval_dataset is not None else None,
        use_weighted_sampler=use_weighted_sampler,
        train_labels=train_labels,
        num_classes=num_classes,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    trainer.train()

    final_metrics = {}
    if eval_dataset is not None:
        final_metrics = trainer.evaluate()
        print("\n[INFO] Evaluation metrics:")
        for k, v in final_metrics.items():
            print(f"{k}: {v}")

        save_metrics_to_json(final_metrics, os.path.join(output_dir, "validation_metrics.json"))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer