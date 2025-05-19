# /home/sysadm/Music/MedXpert/src/pipeline/blip_captioning.py

def generate_blip_captions(image_paths):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    import os
    
    # Load BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    
    captions = []
    
    for img_path in image_paths:
        # Load and process each image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to("cuda")
        
        # Generate caption
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
    
    return captions