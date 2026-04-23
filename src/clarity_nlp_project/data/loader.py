from datasets import load_dataset


def load_clarity_dataset(config):
    dataset_name = config["dataset"]["name"]
    dataset_config_name = config["dataset"].get("config_name", None)

    print(f"[INFO] Loading dataset: {dataset_name}...")

    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name)
    else:
        dataset = load_dataset(dataset_name)

    print("[INFO] Dataset loaded successfully!")
    return dataset


def inspect_dataset(dataset, config):
    print("\n[INFO] Dataset structure:")
    print(dataset)

    print("\n[INFO] Available splits:")
    for split in dataset.keys():
        print(f" - {split}: {len(dataset[split])} samples")

    first_split = list(dataset.keys())[0]
    sample = dataset[first_split][0]

    print("\n[INFO] Sample example:")
    for key, value in sample.items():
        print(f"{key}: {value}")