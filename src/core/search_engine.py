# /home/sysadm/Music/MedXpert/src/core/search_engine.py
import os
import yaml
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load config
def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "config", "clip_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Get absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(base_dir, config["training"]["save_path"])
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")  # Path to your best model
EMBED_PATH = os.path.join(base_dir, config["embeddings"]["save_dir"])

# Load model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model architecture
model = CLIPModel.from_pretrained(config["model"]["name"]).to(device)

# Load your fine-tuned weights from best_model.pt
# Fixed: Properly load state_dict from the checkpoint dictionary
try:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with validation accuracy: {checkpoint.get('val_acc_avg', 'unknown')}")
    else:
        # Try direct loading if not in the expected format
        model.load_state_dict(checkpoint)
        print("Loaded model directly from checkpoint")
        
    model = model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Looking for model at: {BEST_MODEL_PATH}")
    raise

# Load the processor
processor = CLIPProcessor.from_pretrained(config["model"]["name"])

# Load embeddings (which you've already created)
try:
    image_embeddings = torch.load(os.path.join(EMBED_PATH, "train_image.pt")).to(device)
    text_embeddings = torch.load(os.path.join(EMBED_PATH, "train_text.pt")).to(device)
    print(f"Loaded {image_embeddings.shape[0]} image embeddings and {text_embeddings.shape[0]} text embeddings")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    print(f"Looking for embeddings at: {os.path.join(EMBED_PATH, 'train_image.pt')}")
    raise

# Cosine similarity search
def search_text_by_image(query_image_path, k=5):
    """
    Search for text using an image.
    
    Args:
        query_image_path: Path to the image file
        k: Number of results to return
        
    Returns:
        Tuple of (indices, similarity_scores)
    """
    # FIX: Load image from path if a string is provided
    if isinstance(query_image_path, str):
        try:
            query_image = Image.open(query_image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not open image at path {query_image_path}: {e}")
    else:
        # Assume it's already a PIL Image
        query_image = query_image_path
    
    # Process the image with the CLIP processor
    inputs = processor(images=query_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        query_embed = model.get_image_features(**inputs)
        query_embed = F.normalize(query_embed, dim=-1)
        
    db_texts = F.normalize(text_embeddings, dim=-1)
    sims = query_embed @ db_texts.T
    topk = torch.topk(sims.squeeze(0), k)
    return topk.indices.tolist(), topk.values.tolist()

def search_image_by_text(query_text: str, k=5):
    """
    Search for images using a text query.
    
    Args:
        query_text: Text query
        k: Number of results to return
        
    Returns:
        Tuple of (indices, similarity_scores)
    """
    inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        query_embed = model.get_text_features(**inputs)
        query_embed = F.normalize(query_embed, dim=-1)
        
    db_images = F.normalize(image_embeddings, dim=-1)
    sims = query_embed @ db_images.T
    topk = torch.topk(sims.squeeze(0), k)
    return topk.indices.tolist(), topk.values.tolist()
