from transformers import AutoModelForSequenceClassification


def get_model(config, num_labels, id2label, label2id):
    model_name = config["model"]["name"]

    print(f"[INFO] Loading model: {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    return model