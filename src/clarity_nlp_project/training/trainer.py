import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_model(config, model, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=config["paths"]["models_dir"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=config["paths"]["reports_dir"],
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=None,
        compute_metrics=compute_metrics,
    )

    print("\n[INFO] Starting training...")
    trainer.train()

    print("\n[INFO] Training completed!")

    return trainer