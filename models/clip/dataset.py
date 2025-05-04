# import torch
# from torch.utils.data import Dataset
# from PIL import Image

# class MIMICCLIPDataset(Dataset):
#     def __init__(self, hf_dataset, processor, max_length=77):
#         self.dataset = hf_dataset
#         self.processor = processor
#         self.max_length = max_length  # CLIP default context length
        
#     def __getitem__(self, idx):
#         # Get the item from the dataset
#         item = self.dataset[idx]
        
#         # Check if item is a string (which would cause the error)
#         if isinstance(item, str):
#             # Handle the case where the dataset returns strings instead of dictionaries
#             # This might be a symptom of an earlier processing issue
#             raise ValueError(f"Dataset returned a string at index {idx} instead of a dictionary: {item[:100]}...")
        
#         # Make sure image is a PIL Image
#         if "image" not in item:
#             raise KeyError(f"No 'image' key found in item at index {idx}. Keys available: {list(item.keys())}")
        
#         image = item["image"]
#         if not isinstance(image, Image.Image):
#             raise TypeError(f"Expected PIL.Image at index {idx}, got {type(image)} instead")
            
#         # Get text - prefer impression, fall back to findings
#         if "impression" not in item and "findings" not in item:
#             raise KeyError(f"Neither 'impression' nor 'findings' keys found in item at index {idx}")
            
#         text = item.get("impression") or item.get("findings")
#         if not isinstance(text, str):
#             raise TypeError(f"Expected string for text at index {idx}, got {type(text)} instead")
        
#         # Process with CLIP processor
#         try:
#             inputs = self.processor(
#                 text=text,
#                 images=image,
#                 padding="max_length",
#                 max_length=self.max_length,
#                 truncation=True,
#                 return_tensors="pt"
#             )
            
#             # Remove batch dimension added by processor
#             return {k: v.squeeze(0) for k, v in inputs.items()}
            
#         except Exception as e:
#             raise RuntimeError(f"Error processing item {idx}: {str(e)}")
    
#     def __len__(self):
#         return len(self.dataset)
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import os

# class MIMICCLIPDataset(Dataset):
#     def __init__(self, data_dir, split, processor, max_length=77):
#         """
#         Args:
#             data_dir (str): Root directory where 'texts/{split}.json' and 'images/' exist.
#             split (str): One of 'train', 'validation', 'test'.
#             processor: CLIP processor (e.g., CLIPProcessor).
#             max_length (int): Max token length for text. Default is CLIP’s 77.
#         """
#         import json

#         self.processor = processor
#         self.max_length = max_length

#         # Load annotations
#         json_path = os.path.join(data_dir, "texts", f"{split}.json")
#         with open(json_path, "r") as f:
#             self.samples = json.load(f)

#         self.image_root = os.path.join(data_dir, "images")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         # Load image from disk
#         image_path = os.path.join(self.image_root, os.path.basename(sample["image_id"]))
#         try:
#             image = Image.open(image_path).convert("RGB")
#         except Exception as e:
#             raise RuntimeError(f"Failed to load image at {image_path}: {str(e)}")

#         # Text
#         text = sample["text"]
#         if not isinstance(text, str):
#             raise TypeError(f"Expected string for text at index {idx}, got {type(text)}")

#         # Tokenize and preprocess
#         inputs = self.processor(
#             text=text,
#             images=image,
#             padding="max_length",
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors="pt"
#         )

#         return {k: v.squeeze(0) for k, v in inputs.items()}
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class MIMICCLIPDataset(Dataset):
    def __init__(self, data_dir, split, processor, max_length=77):
        """
        Args:
            data_dir (str): Root directory where 'texts/{split}.json' and 'images/' exist.
            split (str): One of 'train', 'validation', 'test'.
            processor: CLIP processor (e.g., CLIPProcessor).
            max_length (int): Max token length for text. Default is CLIP's 77.
        """
        self.processor = processor
        self.max_length = max_length
        
        # Ensure data_dir is absolute
        self.data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        
        # Load annotations
        json_path = os.path.join(self.data_dir, "texts", f"{split}.json")
        print(f"Loading dataset from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset file not found: {json_path}")
            
        with open(json_path, "r") as f:
            self.samples = json.load(f)
            
        print(f"Successfully loaded {len(self.samples)} samples from {split} split")
        
        # Keep track of any missing images for debugging
        self.missing_images = []
        
        # Validate a few paths to help with debugging
        if self.samples:
            first_sample = self.samples[0]
            print(f"\nFirst sample data:")
            print(f"  - image_id: {first_sample.get('image_id', 'N/A')}")
            print(f"  - image_path: {first_sample.get('image_path', 'N/A')}")
            
            # Check if the expected image exists
            if 'image_path' in first_sample:
                img_path = os.path.join(self.data_dir, first_sample['image_path'])
                print(f"  - Full image path: {img_path}")
                print(f"  - Image exists: {os.path.isfile(img_path)}")
            
            # Check image paths for first 5 samples to validate
            print("\nValidating image paths for first 5 samples:")
            for i, sample in enumerate(self.samples[:5]):
                if i >= 5:
                    break
                    
                if 'image_path' in sample:
                    img_path = os.path.join(self.data_dir, sample['image_path'])
                    exists = os.path.isfile(img_path)
                    print(f"  Sample {i}: {os.path.basename(img_path)} - {'✓' if exists else '✗'}")
                else:
                    print(f"  Sample {i}: No image_path field")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = None
        
        # First, try to use image_path if available (preferred method)
        if 'image_path' in sample and sample['image_path']:
            try:
                image_path = os.path.join(self.data_dir, sample['image_path'])
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # Log the error but don't fail yet - try alternative method
                print(f"Warning: Failed to load image using image_path at index {idx}: {e}")
        
        # If that didn't work, try to use image_id
        if image is None and 'image_id' in sample:
            try:
                # Could be a direct filename or relative path
                image_id = sample['image_id']
                
                # First try as a full path in images directory
                image_path = os.path.join(self.data_dir, "images", image_id)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                else:
                    # Try just the basename in case image_id contains directory info
                    image_path = os.path.join(self.data_dir, "images", os.path.basename(image_id))
                    image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # Now we've exhausted our options, so this is a real error
                if idx not in self.missing_images:
                    self.missing_images.append(idx)
                    if len(self.missing_images) <= 5:  # Limit error logging
                        print(f"Error: Failed to load image for sample {idx}: {e}")
                        print(f"Sample data: {sample}")
                
                # Return a placeholder image as a last resort to avoid breaking the dataloader
                # This is better than crashing but you should investigate these failures
                try:
                    # Create a blank 224x224 RGB image (CLIP's standard size)
                    image = Image.new('RGB', (224, 224), color='gray')
                except:
                    raise RuntimeError(f"Failed to load image for sample {idx} and could not create placeholder")
        
        # Text processing
        text = sample.get("text", "")
        if not isinstance(text, str) or not text:
            print(f"Warning: Invalid text at index {idx}, using empty string")
            text = ""
        
        # Process with CLIP processor
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Periodically report missing images
        if len(self.missing_images) > 0 and len(self.missing_images) % 10 == 0:
            print(f"Warning: {len(self.missing_images)} images missing so far")
        
        return {k: v.squeeze(0) for k, v in inputs.items()}