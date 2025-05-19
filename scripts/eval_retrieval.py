# # scripts/eval_retrieval.py

# import torch
# import yaml
# import os
# from sklearn.metrics import average_precision_score
# from sklearn.metrics.pairwise import cosine_similarity

# # Load config
# with open("config/clip_config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# EMBED_DIR = config["embeddings"]["save_dir"]

# # Load embeddings
# image_embeds = torch.load(os.path.join(EMBED_DIR, "test_image.pt"))
# text_embeds = torch.load(os.path.join(EMBED_DIR, "test_text.pt"))

# # Normalize
# image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
# text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

# # Compute similarity matrix
# sim_matrix = cosine_similarity(text_embeds, image_embeds)

# # Evaluate recall at K
# def recall_at_k(similarity_matrix, k):
#     correct = 0
#     for i in range(len(similarity_matrix)):
#         top_k = similarity_matrix[i].argsort()[::-1][:k]
#         if i in top_k:
#             correct += 1
#     return correct / len(similarity_matrix)

# if __name__ == "__main__":
#     for k in [1, 5, 10]:
#         r_at_k = recall_at_k(sim_matrix, k)
#         print(f"🔍 Recall@{k}: {r_at_k:.4f}")
#     print("🔍 Evaluation complete.")

# scripts/eval_retrieval.py
import torch
import yaml
import os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Load config
def load_config():
    config_path = os.path.join(project_root, "config", "clip_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

try:
    config = load_config()
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Try an alternative path
    alternative_path = os.path.join(script_dir, "..", "config", "clip_config.yaml")
    print(f"Trying alternative path: {alternative_path}")
    if os.path.exists(alternative_path):
        with open(alternative_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Alternative path not found either. Available directories in project root:")
        for item in os.listdir(project_root):
            if os.path.isdir(os.path.join(project_root, item)):
                print(f"- {item}")
        raise

# Get the correct path for embeddings
EMBED_DIR = os.path.join(project_root, config["embeddings"]["save_dir"])
print(f"Looking for embeddings in: {EMBED_DIR}")

# Load embeddings
try:
    image_embeds_path = os.path.join(EMBED_DIR, "test_image.pt")
    text_embeds_path = os.path.join(EMBED_DIR, "test_text.pt")
    
    if not os.path.exists(image_embeds_path):
        print(f"Warning: Image embeddings not found at {image_embeds_path}")
        print(f"Available files in embedding directory:")
        if os.path.exists(EMBED_DIR):
            for file in os.listdir(EMBED_DIR):
                print(f"- {file}")
        else:
            print(f"Directory {EMBED_DIR} does not exist")
    
    image_embeds = torch.load(image_embeds_path)
    text_embeds = torch.load(text_embeds_path)
    print(f"Successfully loaded embeddings. Shapes: Image {image_embeds.shape}, Text {text_embeds.shape}")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    raise

# Convert to numpy for sklearn compatibility
image_embeds_np = image_embeds.cpu().numpy()
text_embeds_np = text_embeds.cpu().numpy()

# Normalize embeddings
image_embeds_np = image_embeds_np / np.linalg.norm(image_embeds_np, axis=1, keepdims=True)
text_embeds_np = text_embeds_np / np.linalg.norm(text_embeds_np, axis=1, keepdims=True)

# Compute similarity matrix
sim_matrix = np.dot(text_embeds_np, image_embeds_np.T)

# Evaluate recall at K for text->image retrieval
def recall_at_k(similarity_matrix, k):
    n = similarity_matrix.shape[0]
    correct = 0
    
    # Assuming diagonal elements are the ground truth matches
    # (i.e., text_i corresponds to image_i)
    for i in range(n):
        # Get top k indices for this text query
        top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]
        # Check if the correct image (index i) is in the top k
        if i in top_k_indices:
            correct += 1
            
    return correct / n

# Also evaluate image->text retrieval
def image_to_text_recall_at_k(similarity_matrix, k):
    # Transpose to get image->text similarities
    sim_matrix_t = similarity_matrix.T
    n = sim_matrix_t.shape[0]
    correct = 0
    
    for i in range(n):
        top_k_indices = np.argsort(sim_matrix_t[i])[::-1][:k]
        if i in top_k_indices:
            correct += 1
            
    return correct / n

if __name__ == "__main__":
    print("Text-to-Image Retrieval:")
    for k in [1, 5, 10]:
        r_at_k = recall_at_k(sim_matrix, k)
        print(f"🔍 Recall@{k}: {r_at_k:.4f}")
        
    print("\nImage-to-Text Retrieval:")
    for k in [1, 5, 10]:
        r_at_k = image_to_text_recall_at_k(sim_matrix, k)
        print(f"🔍 Recall@{k}: {r_at_k:.4f}")
        
    print("🔍 Evaluation complete.")