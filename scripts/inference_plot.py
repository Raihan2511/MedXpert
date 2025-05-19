# scripts/inference_plot.py

import os
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

# Load config
def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "clip_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Get absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(base_dir, config["training"]["save_path"])
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")  # Path to your best model
EMBED_DIR = os.path.join(base_dir, config["embeddings"]["save_dir"])
IMAGE_DIR = os.path.join(base_dir, "data/processed/images")
TOP_K = 5

# Load model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model architecture
model = CLIPModel.from_pretrained(config["model"]["name"]).to(device)
# Load your fine-tuned weights from best_model.pt
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with validation accuracy: {checkpoint.get('val_acc_avg', 'unknown')}")
else:
    # Try direct loading if not in the expected format
    model.load_state_dict(checkpoint)
    print("Loaded model directly from checkpoint")

model = model.eval()  # Set model to evaluation mode

# Load the processor
processor = CLIPProcessor.from_pretrained(config["model"]["name"])

# Load embeddings
text_feats = torch.load(os.path.join(EMBED_DIR, "test_text.pt"))
image_feats = torch.load(os.path.join(EMBED_DIR, "test_image.pt"))

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
        os.makedirs(os.path.join(base_dir, "results/inference_plots"), exist_ok=True)
        save_path = os.path.join(base_dir, f"results/inference_plots/query_{query_idx}.png")
        plt.savefig(save_path)
        print(f"✅ Saved to {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    plot_topk(query_idx=0, save=True)