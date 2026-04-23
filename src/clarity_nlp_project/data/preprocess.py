def clean_text(text, config):
    if text is None:
        text = ""

    text = str(text)

    if config["preprocessing"]["lowercase"]:
        text = text.lower()

    if config["preprocessing"]["remove_extra_spaces"]:
        text = " ".join(text.split())

    return text


def build_label_mapping(dataset, label_column):
    labels = sorted(set(dataset["train"][label_column]))

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print("\n[INFO] Label mapping:")
    for label, idx in label2id.items():
        print(f"{label} -> {idx}")

    return label2id, id2label


def preprocess_dataset(dataset, config):
    label_col = config["dataset"]["label_column"]

    label2id, id2label = build_label_mapping(dataset, label_col)

    def preprocess_example(example):
        question = example.get("question", "")
        answer = example.get("interview_answer", "")

        combined_text = f"Question: {question}\n\nAnswer: {answer}"
        cleaned_text = clean_text(combined_text, config)

        encoded_label = label2id[example[label_col]]

        return {
            "text": cleaned_text,
            "label": encoded_label,
        }

    processed_dataset = dataset.map(preprocess_example)

    columns_to_keep = ["text", "label"]

    for split in processed_dataset.keys():
        cols_to_remove = [
            col for col in processed_dataset[split].column_names
            if col not in columns_to_keep
        ]
        processed_dataset[split] = processed_dataset[split].remove_columns(cols_to_remove)

    return processed_dataset, label2id, id2label