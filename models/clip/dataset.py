# models/clip/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image

class MIMICCLIPDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=77):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length  # CLIP default context length
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # already a PIL.Image from dataset
        
        # Preferentially use impression if available, otherwise findings
        text = item.get("impression") or item.get("findings")
        
        # Process with CLIP processor 
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    def __len__(self):
        return len(self.dataset)

# Example of how to create and use the DataLoader with your dataset
'''
import torch
from torch.utils.data import DataLoader

# Create dataset
train_dataset = MIMICCLIPDataset(train_hf_dataset, processor)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16, 
    shuffle=True,
    num_workers=4
)

# Example training loop
for batch, _ in train_loader:
    # batch is now a dict with keys like 'input_ids', 'attention_mask', 'pixel_values'
    # that can be passed directly to your CLIP model
    outputs = model(**batch)
    loss = outputs.loss
    # ... rest of training code
'''
