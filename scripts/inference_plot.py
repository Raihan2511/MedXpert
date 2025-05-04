# scripts/inference_plot.py

import os
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

# Load config
with open("config/clip_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["training"]["save_path"]
EMBED_DIR = config["embeddings"]["save_dir"]
IMAGE_DIR = "data/processed/images"
TOP_K = 5

# Load raw text queries if available (e.g., from dataset)


# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)
model.eval()

# Load embeddings
text_feats = torch.load(f"{EMBED_DIR}/test_text.pt")
image_feats = torch.load(f"{EMBED_DIR}/test_image.pt")

# Load all test image paths
test_image_paths = sorted([
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.startswith("test") and fname.endswith(".jpg")
])

# Similarity matrix
sim_matrix = cosine_similarity(text_feats, image_feats)
# with open("data/processed/test_texts.txt", "r") as f:
#     raw_texts = [line.strip() for line in f]

# Plot top-k
def plot_topk(query_idx, save=False):
    sims = sim_matrix[query_idx]
    topk = sims.argsort()[::-1][:TOP_K]

    plt.figure(figsize=(15, 5))
    for rank, idx in enumerate(topk):
        image = Image.open(test_image_paths[idx])
        plt.subplot(1, TOP_K, rank + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Rank {rank+1}\nSim: {sims[idx]:.2f}")

    plt.suptitle(f"Top-{TOP_K} Retrieved Images for Query {query_idx}", fontsize=16)
#     plt.suptitle(
#     f"Top-{TOP_K} Retrieved Images\nQuery {query_idx}: {raw_texts[query_idx]}",
#     fontsize=12
# )

    plt.tight_layout()

    if save:
        os.makedirs("results/inference_plots", exist_ok=True)
        plt.savefig(f"results/inference_plots/query_{query_idx}.png")
        print(f"✅ Saved to results/inference/plots/query_{query_idx}.png")

    plt.show()


# Example usage
if __name__ == "__main__":
    plot_topk(query_idx=0, save=True)

