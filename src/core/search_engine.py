# src/core/search_engine.py

import os
import yaml
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load config
def load_config():
    with open("config/clip_config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
MODEL_PATH = config["training"]["save_path"]
EMBED_PATH = config["embeddings"]["save_dir"]

# Load model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_PATH).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

# Load embeddings
# image_embeddings = torch.load(os.path.join(EMBED_PATH, "train_image.pt"))
# text_embeddings = torch.load(os.path.join(EMBED_PATH, "train_text.pt"))
image_embeddings = torch.load(os.path.join(EMBED_PATH, "train_image.pt")).to(device)
text_embeddings = torch.load(os.path.join(EMBED_PATH, "train_text.pt")).to(device)


# Cosine similarity search
def search_text_by_image(query_image: Image.Image, k=5):
    inputs = processor(images=query_image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embed = model.get_image_features(**inputs)

    query_embed = F.normalize(query_embed, dim=-1)
    db_texts = F.normalize(text_embeddings, dim=-1)
    sims = query_embed @ db_texts.T
    topk = torch.topk(sims.squeeze(0), k)
    return topk.indices.tolist(), topk.values.tolist()

def search_image_by_text(query_text: str, k=5):
    inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embed = model.get_text_features(**inputs)

    query_embed = F.normalize(query_embed, dim=-1)
    db_images = F.normalize(image_embeddings, dim=-1)
    sims = query_embed @ db_images.T
    topk = torch.topk(sims.squeeze(0), k)
    return topk.indices.tolist(), topk.values.tolist()
