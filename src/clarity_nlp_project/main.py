import yaml

from .data.loader import load_clarity_dataset, inspect_dataset
from .data.preprocess import preprocess_dataset
from .data.tokenizer_utils import tokenize_dataset
from .models.hf_classifier import get_model
from .training.trainer import train_model


def load_config(path="configs/default.yaml"):
    """
    Carica il file di configurazione YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Pipeline principale:
    1. carica config
    2. carica dataset
    3. ispeziona dataset
    4. preprocessa testo e label
    5. tokenizza
    6. crea il modello
    7. addestra il modello
    """
    print("[INFO] Loading configuration...")
    config = load_config()
    print("[INFO] Configuration loaded successfully!")

    # 1. Load dataset
    dataset = load_clarity_dataset(config)

    # 2. Inspect dataset
    inspect_dataset(dataset, config)

    # 3. Preprocess dataset
    print("\n[INFO] Starting preprocessing...")
    processed_dataset, label2id, id2label = preprocess_dataset(dataset, config)
    print("[INFO] Preprocessing completed!")

    print("\n[INFO] Sample after preprocessing:")
    print(processed_dataset["train"][0])

    # 4. Tokenize dataset
    print("\n[INFO] Starting tokenization...")
    tokenized_dataset = tokenize_dataset(processed_dataset, config)
    print("[INFO] Tokenization completed!")

    print("\n[INFO] Sample after tokenization:")
    print(tokenized_dataset["train"][0])

    print("\n[INFO] Final dataset summary:")
    print(tokenized_dataset)

    print("\n[INFO] label2id:")
    print(label2id)

    print("\n[INFO] id2label:")
    print(id2label)

    # 5. Build model
    print("\n[INFO] Building model...")
    num_labels = len(label2id)
    model = get_model(config, num_labels, id2label, label2id)
    print("[INFO] Model loaded successfully!")

    # 6. Train model
    trainer = train_model(config, model, tokenized_dataset)

    print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main()