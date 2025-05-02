# # scripts/preprocess_data.py

# import os
# import json
# from datasets import load_dataset
# from tqdm import tqdm

# # Configs
# DATASET_NAME = "itsanmolgupta/mimic-cxr-dataset"
# OUTPUT_DIR = "data/processed/"
# MIN_TEXT_LENGTH = 8

# os.makedirs(os.path.join(OUTPUT_DIR, "texts"), exist_ok=True)

# # Load dataset
# print("Loading dataset from Hugging Face...")
# dataset = load_dataset(DATASET_NAME)

# # Clean and filter function
# def filter_and_format(example):
#     text = example.get("impression") or example.get("findings")
#     if not text or len(text.strip()) < MIN_TEXT_LENGTH:
#         return None
#     return {
#         "image_id": example["image_id"],
#         "text": text.strip().replace("\n", " ")
#     }

# # Apply filtering
# for split in ["train", "validation", "test"]:
#     print(f"Processing {split} split...")
#     processed = []
#     for item in tqdm(dataset[split]):
#         record = filter_and_format(item)
#         if record:
#             processed.append(record)

#     with open(os.path.join(OUTPUT_DIR, "texts", f"{split}.json"), "w") as fout:
#         json.dump(processed, fout, indent=2)

# print("✅ Preprocessing complete. Cleaned JSONs written to data/processed/texts/")

# scripts/preprocess_data.py

import os
import json
from tqdm import tqdm

MIN_TEXT_LENGTH = 8

# Clean and filter function
def filter_and_format(example):
    text = example.get("impression") or example.get("findings")
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None
    return {
        "image_id": example["image_id"],
        "text": text.strip().replace("\n", " ")
    }

def preprocess_dataset(dataset, output_dir):
    os.makedirs(os.path.join(output_dir, "texts"), exist_ok=True)
    for split in ["train", "validation", "test"]:
        print(f"Processing {split} split...")
        processed = []
        for item in tqdm(dataset[split]):
            record = filter_and_format(item)
            if record:
                processed.append(record)

        with open(os.path.join(output_dir, "texts", f"{split}.json"), "w") as fout:
            json.dump(processed, fout, indent=2)

    print(f"✅ Preprocessing complete. Cleaned JSONs written to {output_dir}/texts/")

