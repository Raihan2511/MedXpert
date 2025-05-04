# scripts/eval_retrieval.py

import torch
import yaml
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# Load config
with open("config/clip_config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBED_DIR = config["embeddings"]["save_dir"]

# Load embeddings
image_embeds = torch.load(os.path.join(EMBED_DIR, "test_image.pt"))
text_embeds = torch.load(os.path.join(EMBED_DIR, "test_text.pt"))

# Normalize
image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

# Compute similarity matrix
sim_matrix = cosine_similarity(text_embeds, image_embeds)

# Evaluate recall at K
def recall_at_k(similarity_matrix, k):
    correct = 0
    for i in range(len(similarity_matrix)):
        top_k = similarity_matrix[i].argsort()[::-1][:k]
        if i in top_k:
            correct += 1
    return correct / len(similarity_matrix)

if __name__ == "__main__":
    for k in [1, 5, 10]:
        r_at_k = recall_at_k(sim_matrix, k)
        print(f"🔍 Recall@{k}: {r_at_k:.4f}")
    print("🔍 Evaluation complete.")