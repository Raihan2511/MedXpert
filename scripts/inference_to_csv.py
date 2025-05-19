# scripts/inference_to_csv.py

import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load config
def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "config", "clip_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(base_dir, config["training"]["save_path"])
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")  # Path to your best model
EMBED_PATH = os.path.join(base_dir, config["embeddings"]["save_dir"])
IMAGE_DIR = "data/processed/images"
TOP_K = 5
CSV_PATH = "results/inference_plots/inference_topk.csv"

# Load embeddings
text_feats = torch.load(f"{EMBED_PATH}/test_text.pt")
image_feats = torch.load(f"{EMBED_PATH}/test_image.pt")

# List image files
test_image_paths = sorted([
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.startswith("test") and fname.endswith(".jpg")
])

# Compute cosine similarity
sim_matrix = cosine_similarity(text_feats, image_feats)

# Extract top-k results
rows = []
for i, sims in enumerate(tqdm(sim_matrix, desc="🔍 Computing top-k")):
    topk = sims.argsort()[::-1][:TOP_K]
    for rank, idx in enumerate(topk):
        rows.append({
            "query_index": i,
            "rank": rank + 1,
            "image_path": test_image_paths[idx],
            "similarity": sims[idx]
        })

# Save to CSV
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
print(f"✅ Saved Top-{TOP_K} similarity results to {CSV_PATH}")