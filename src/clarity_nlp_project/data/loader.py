from datasets import load_dataset


def load_clarity_dataset(config):
    """
    Carica il dataset CLARITY usando Hugging Face.
    """
    dataset_name = config["dataset"]["name"]

    print(f"[INFO] Loading dataset: {dataset_name}...")

    dataset = load_dataset(dataset_name)

    print("[INFO] Dataset loaded successfully!")

    return dataset


def inspect_dataset(dataset, config):
    """
    Stampa informazioni utili sul dataset.
    """
    text_col = config["dataset"]["text_column"]
    label_col = config["dataset"]["label_column"]

    print("\n[INFO] Dataset structure:")
    print(dataset)

    print("\n[INFO] Available splits:")
    for split in dataset.keys():
        print(f" - {split}: {len(dataset[split])} samples")

    # Mostra un esempio
    first_split = list(dataset.keys())[0]
    sample = dataset[first_split][0]

    print("\n[INFO] Sample example:")
    for key, value in sample.items():
        print(f"{key}: {value}")

    # Controllo colonne
    if text_col not in sample:
        print(f"\n[WARNING] '{text_col}' not found in dataset!")
    if label_col not in sample:
        print(f"[WARNING] '{label_col}' not found in dataset!")