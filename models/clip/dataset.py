# models/clip/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image

class MIMICCLIPDataset(Dataset):
    """
    Dataset class for MIMIC-CXR for use with CLIPProcessor.
    Accepts a Hugging Face dataset split.
    """
    def __init__(self, hf_dataset, processor, use_findings_if_missing=True):
        self.dataset = hf_dataset
        self.processor = processor
        self.use_findings = use_findings_if_missing

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # PIL.Image.Image from Hugging Face
        text = item.get("impression") or (item.get("findings") if self.use_findings else "")

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}, None

    def __len__(self):
        return len(self.dataset)
