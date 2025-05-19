import os
import sys
import yaml
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader

# Add parent directory to path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.clip.dataset import CLIPDataset
from models.clip.dataset import ProcessedCLIPDataset

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
config_path = os.path.join(project_dir, "config", "clip_config.yaml")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

EMBED_DIR = config["embeddings"].get("save_dir", "data/embeddings")
BATCH_SIZE = config["training"].get("batch_size", 16)

os.makedirs(EMBED_DIR, exist_ok=True)

def encode_and_save(model, processor, device, split_name: str, dataset):
    print(f"\n🔄 Encoding {split_name} split")
    mimic_ds = CLIPDataset(dataset)

    # Use the same collate_fn pattern as in main.py
    def clip_collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        return inputs

    dataloader = DataLoader(mimic_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=clip_collate_fn)
    image_embeds = []
    text_embeds = []

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            image_feat = model.get_image_features(pixel_values=batch["pixel_values"])
            text_feat = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        image_embeds.append(image_feat.cpu())
        text_embeds.append(text_feat.cpu())

    torch.save(torch.cat(image_embeds), os.path.join(EMBED_DIR, f"{split_name}_image.pt"))
    torch.save(torch.cat(text_embeds), os.path.join(EMBED_DIR, f"{split_name}_text.pt"))
    print(f"✅ Saved: {split_name}_image.pt and {split_name}_text.pt")

def save_all_embeddings(model, processor, device, dataset_dict):
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset_dict:
            encode_and_save(model, processor, device, split_name, dataset_dict[split_name])