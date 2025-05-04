# # scripts/save_embeddings.py

# import os
# import yaml
# import torch
# from tqdm import tqdm
# from datasets import load_dataset
# from transformers import CLIPModel, CLIPProcessor
# from torch.utils.data import DataLoader
# import sys
# sys.path.append(".")
# from models.clip.dataset import MIMICCLIPDataset



# # Load config
# with open("config/clip_config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# MODEL_PATH = config["training"]["save_path"]
# DATASET_NAME = config["dataset"]["name"]
# USE_FINDINGS = config["dataset"].get("use_findings_if_missing", True)
# EMBED_DIR = config["embeddings"]["save_dir"]
# BATCH_SIZE = config["training"]["batch_size"]
# SPLITS = ["train", "validation", "test"]

# os.makedirs(EMBED_DIR, exist_ok=True)

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
# processor = CLIPProcessor.from_pretrained(MODEL_PATH)
# model.eval()

# # Save features for each split
# def encode_and_save(split_name):
#     print(f"\n🔄 Encoding {split_name} split")
#     dataset = load_dataset(DATASET_NAME, split=split_name)
#     mimic_ds = MIMICCLIPDataset(dataset, processor)
#     dataloader = DataLoader(mimic_ds, batch_size=BATCH_SIZE)

#     image_embeds = []
#     text_embeds = []

#     for batch, _ in tqdm(dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             image_feat = model.get_image_features(batch["pixel_values"])
#             text_feat = model.get_text_features(batch["input_ids"], attention_mask=batch["attention_mask"])

#         image_embeds.append(image_feat.cpu())
#         text_embeds.append(text_feat.cpu())

#     torch.save(torch.cat(image_embeds), f"{EMBED_DIR}/{split_name}_image.pt")
#     torch.save(torch.cat(text_embeds), f"{EMBED_DIR}/{split_name}_text.pt")
#     print(f"✅ Saved embeddings to {EMBED_DIR}/{split_name}_*.pt")

# if __name__ == "__main__":
#     for split in SPLITS:
#         encode_and_save(split)
# scripts/save_embeddings.py

import os
import yaml
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from models.clip.dataset import MIMICCLIPDataset  # assumes project structure is correct

# Load config
with open("config/clip_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = config["dataset"]["root"]  # new: path to your /images and /texts folders
MODEL_PATH = config["training"]["save_path"]
EMBED_DIR = config["embeddings"]["save_dir"]
BATCH_SIZE = config["training"]["batch_size"]
SPLITS = ["train", "validation", "test"]

# Ensure output directory exists
os.makedirs(EMBED_DIR, exist_ok=True)

# Setup device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)
model.eval()

def encode_and_save(split):
    print(f"\n🔄 Encoding '{split}' split")
    dataset = MIMICCLIPDataset(DATA_DIR, split, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    image_embeds, text_embeds = [], []

    for batch in tqdm(dataloader, desc=f"Encoding {split}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            img_feat = model.get_image_features(batch["pixel_values"])
            txt_feat = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        image_embeds.append(img_feat.cpu())
        text_embeds.append(txt_feat.cpu())

    torch.save(torch.cat(image_embeds), os.path.join(EMBED_DIR, f"{split}_image.pt"))
    torch.save(torch.cat(text_embeds), os.path.join(EMBED_DIR, f"{split}_text.pt"))
    print(f"✅ Saved: {split}_image.pt and {split}_text.pt")

if __name__ == "__main__":
    for split in SPLITS:
        encode_and_save(split)

