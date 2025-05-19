# # # scripts/preprocess_data.py
# import os
# import json
# from tqdm import tqdm

# MIN_TEXT_LENGTH = 10  # Minimum length of text to consider valid

# def filter_and_format(example):
#     text = example.get("impression") or example.get("findings")
#     if not text or len(text.strip()) < MIN_TEXT_LENGTH:
#         return None
    
#     # Use 'image_path' as the image identifier if available, otherwise use an auto-generated ID
#     image_id = example.get("image_path", f"image_{hash(text)}")
    
#     return {
#         "image_id": image_id,
#         "text": text.strip().replace("\n", " ")
#     }

# def preprocess_dataset(dataset, output_dir):
#     # Handle split if only 'train' exists
#     if "validation" not in dataset or "test" not in dataset:
#         print("\nℹ️ No validation/test split found. Splitting train set into train/val/test...")
        
#         dataset["train"] = dataset["train"].shuffle(seed=42)
#         total = len(dataset["train"])
#         train_end = int(total * 0.8)
#         val_end = train_end + int(total * 0.1)
        
#         dataset["validation"] = dataset["train"].select(range(train_end, val_end))
#         dataset["test"] = dataset["train"].select(range(val_end, total))
#         dataset["train"] = dataset["train"].select(range(0, train_end))
    
#     # Create output directories
#     os.makedirs(os.path.join(output_dir, "texts"), exist_ok=True)
#     image_dir = os.path.join(output_dir, "images")
#     os.makedirs(image_dir, exist_ok=True)
    
#     # Process each split
#     for split in ["train", "validation", "test"]:
#         print(f"Processing {split} split...")
#         processed = []
        
#         for i, item in enumerate(tqdm(dataset[split])):
#             # Create a serializable version of the item
#             serializable_item = {}
            
#             # Process each key in the item
#             for key, value in item.items():
#                 # Handle PIL image objects
#                 if hasattr(value, "save") and callable(value.save):  # Check if it's a PIL image
#                     image_filename = f"{split}_{i:05d}_{key}.jpg"
#                     image_path = os.path.join(image_dir, image_filename)
#                     value.save(image_path)
                    
#                     # Store the path instead of the image object
#                     serializable_item[f"{key}_path"] = f"images/{image_filename}"
#                 else:
#                     # Keep non-image data as is
#                     serializable_item[key] = value
            
#             # Process the item with only serializable content
#             record = filter_and_format(serializable_item)
            
#             if record:
#                 processed.append(record)
        
#         # Save the processed text data
#         with open(os.path.join(output_dir, "texts", f"{split}.json"), "w") as fout:
#             json.dump(processed, fout, indent=2)
        
#         print(f"Saved {len(processed)} records for {split} split")

import os
import json
from tqdm import tqdm

MIN_TEXT_LENGTH = 10  # Minimum length of text to consider valid

# def filter_and_format(example):
#     """
#     Filter and format examples for CLIP dataset.
    
#     Args:
#         example (dict): Input example with 'findings', 'impression', etc.
        
#     Returns:
#         dict or None: Formatted example or None if invalid
#     """
#     # Use impression if available, otherwise findings
#     text = example.get("impression") or example.get("findings")
#     if not text or len(text.strip()) < MIN_TEXT_LENGTH:
#         return None
    
#     # Make sure we have a valid image_id that matches the actual image filename
#     image_id = example.get("image_id")
#     if not image_id:
#         return None
    
#     return {
#         "image_id": image_id,
#         "image_path": example.get("image_path"),  # Store the image path explicitly
#         "text": text.strip().replace("\n", " ")
#     }
def filter_and_format(example):
    """
    Filter and format examples for CLIP dataset.
    
    Args:
        example (dict): Input example with 'findings', 'impression', etc.
        
    Returns:
        dict or None: Formatted example or None if invalid
    """
    # Only use impression (not findings)
    text = example.get("impression")
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None
    
    # Make sure we have a valid image_id that matches the actual image filename
    image_id = example.get("image_id")
    if not image_id:
        return None
    
    return {
        "image_id": image_id,
        "image_path": example.get("image_path"),  # Store the image path explicitly
        "text": text.strip().replace("\n", " ")
    }
def preprocess_dataset(dataset, output_dir):
    """
    Preprocess dataset and save text/image data to disk.

    Args:
        dataset (Dataset): HuggingFace dataset object
        output_dir (str): Directory to save processed data
    """
    # Handle split if only 'train' exists
    if "validation" not in dataset or "test" not in dataset:
        print("\nℹ️ No validation/test split found. Splitting train set into train/val/test...")
        
        dataset["train"] = dataset["train"].shuffle(seed=42)
        total = len(dataset["train"])
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        
        dataset["validation"] = dataset["train"].select(range(train_end, val_end))
        dataset["test"] = dataset["train"].select(range(val_end, total))
        dataset["train"] = dataset["train"].select(range(0, train_end))
    
    # Create output directories
    texts_dir = os.path.join(output_dir, "texts")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"Processing dataset to {output_dir}...")
    print(f"Text output directory: {texts_dir}")
    print(f"Image output directory: {image_dir}")
    
    # Process each split
    for split in ["train", "validation", "test"]:
        print(f"\nProcessing {split} split...")
        processed = []
        
        for i, item in enumerate(tqdm(dataset[split])):
            # Create a serializable version of the item
            serializable_item = {}
            
            # Copy the text fields directly
            for key in ["findings", "impression"]:
                if key in item:
                    serializable_item[key] = item[key]
            
            # Process each key in the item that might contain images
            for key, value in item.items():
                # Handle PIL image objects
                if hasattr(value, "save") and callable(value.save):  # Check if it's a PIL image
                    image_filename = f"{split}_{i:05d}_{key}.jpg"
                    full_image_path = os.path.join(image_dir, image_filename)
                    relative_image_path = os.path.join("images", image_filename)
                    
                    try:
                        value.save(full_image_path)
                        print(f"Saved image {i} to {full_image_path}") if i % 1000 == 0 else None
                    except Exception as e:
                        print(f"Failed to save image {i}: {e}")
                        continue
                    
                    # Store both the path and use the filename as the image_id
                    serializable_item["image_path"] = relative_image_path
                    serializable_item["image_id"] = image_filename
            
            # Process the item with only serializable content
            record = filter_and_format(serializable_item)
            
            if record:
                processed.append(record)
        
        # Save the processed text data
        json_path = os.path.join(texts_dir, f"{split}.json")
        with open(json_path, "w") as fout:
            json.dump(processed, fout, indent=2)
        
        print(f"✅ Saved {len(processed)} records for {split} split to {json_path}")
        
        # Verify the first few samples
        if processed:
            print("\nSample data verification:")
            for idx, sample in enumerate(processed[:2]):  # Check first 2 samples
                image_path = os.path.join(output_dir, sample.get("image_path", ""))
                print(f"Sample {idx}:")
                print(f"  - image_id: {sample.get('image_id')}")
                print(f"  - image_path: {sample.get('image_path')}")
                print(f"  - image exists: {os.path.isfile(image_path)}")
                print(f"  - text preview: {sample.get('text', '')[:50]}...")