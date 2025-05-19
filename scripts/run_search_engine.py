# scripts/run_search_engine.py

from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.search_engine import search_image_by_text, search_text_by_image
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



# TEXT -> IMAGE
query_text = "Chest X-ray showing pneumonia"
indices, scores = search_image_by_text(query_text, k=5)
print(f"\n🔍 Results for text query: '{query_text}'")
for i, (idx, score) in enumerate(zip(indices, scores)):
    print(f"Rank {i+1}: Image Index {idx}, Similarity Score: {score:.4f}")

# IMAGE -> TEXT query
# Use correct path - either use absolute path or correct relative path
image_path = os.path.join(project_root, "data", "processed", "images", "test_00001_image.jpg")
# Check if file exists, if not, try to list available images
if not os.path.exists(image_path):
    print(f"Warning: Image not found at {image_path}")
    # Try to find any image in the directory
    img_dir = os.path.join(project_root, "data", "processed", "images")
    if os.path.exists(img_dir):
        available_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if available_images:
            print(f"Available images in directory: {available_images[:5]}")
            # Use the first available image instead
            image_path = os.path.join(img_dir, available_images[0])
            print(f"Using alternative image: {available_images[0]}")
        else:
            print(f"No images found in {img_dir}")
    else:
        print(f"Directory not found: {img_dir}")

# Try to open the image (either original path or alternative)
try:
    query_image = Image.open(image_path)
    indices, scores = search_text_by_image(query_image, k=5)
    print(f"\n🖼️ Results for image query: '{os.path.basename(image_path)}'")
    for i, (idx, score) in enumerate(zip(indices, scores)):
        print(f"Rank {i+1}: Text Index {idx}, Similarity Score: {score:.4f}")
except FileNotFoundError as e:
    print(f"Error: Could not open image file: {e}")
    print("Please check the image path and ensure the file exists.")