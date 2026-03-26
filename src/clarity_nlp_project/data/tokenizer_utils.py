from transformers import AutoTokenizer


def get_tokenizer(config):
    model_name = config["tokenizer"]["model_name"]

    print(f"[INFO] Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def tokenize_dataset(dataset, config):
    tokenizer = get_tokenizer(config)

    max_length = config["tokenizer"]["max_length"]

    def tokenize_example(example):
        return tokenizer(
            example["text"],
            padding=config["tokenizer"]["padding"],
            truncation=config["tokenizer"]["truncation"],
            max_length=max_length
        )

    print("\n[INFO] Tokenizing dataset...")

    tokenized_dataset = dataset.map(tokenize_example, batched=True)

    print("[INFO] Tokenization completed!")

    return tokenized_dataset