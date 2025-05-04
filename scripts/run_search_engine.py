# scripts/run_search_engine.py

from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.search_engine import search_image_by_text, search_text_by_image


# TEXT -> IMAGE
query_text = "Chest X-ray showing pneumonia"
indices, scores = search_image_by_text(query_text, k=5)
print(f"\n🔍 Results for text query: '{query_text}'")
for i, (idx, score) in enumerate(zip(indices, scores)):
    print(f"Rank {i+1}: Image Index {idx}, Similarity Score: {score:.4f}")

# IMAGE -> TEXT
query_image = Image.open("data/processed/images/train_00001.jpg")
indices, scores = search_text_by_image(query_image, k=5)
print(f"\n🖼️ Results for image query: 'train_0001.jpg'")
for i, (idx, score) in enumerate(zip(indices, scores)):
    print(f"Rank {i+1}: Text Index {idx}, Similarity Score: {score:.4f}")
