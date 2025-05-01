# main.py

import os
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
from models.clip.dataset import MIMICCLIPDataset
from models.clip.train import do_train, do_eval

# Config
MODEL_NAME = "openai/clip-vit-base-patch32"
DATASET_NAME = "itsanmolgupta/mimic-cxr-dataset"
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5
SAVE_PATH = "models/clip/fine_tuned"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    print("Loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME)
    train_ds = MIMICCLIPDataset(raw_dataset["train"], processor)
    val_ds = MIMICCLIPDataset(raw_dataset["validation"], processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        train_loss = do_train(model, train_dl, optimizer, lr_scheduler, device)
        val_loss, val_acc = do_eval(model, eval_dl, device)

        print(f"✅ Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
# main.py

import os
import yaml
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
from models.clip.dataset import MIMICCLIPDataset
from models.clip.train import do_train, do_eval

# Load config
with open("config/clip_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
DATASET_NAME = config["dataset"]["name"]
USE_FINDINGS = config["dataset"].get("use_findings_if_missing", True)
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["learning_rate"]
SAVE_PATH = config["training"]["save_path"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    print("Loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME)
    train_ds = MIMICCLIPDataset(raw_dataset["train"], processor, use_findings_if_missing=USE_FINDINGS)
    val_ds = MIMICCLIPDataset(raw_dataset["validation"], processor, use_findings_if_missing=USE_FINDINGS)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        train_loss = do_train(model, train_dl, optimizer, lr_scheduler, device)
        val_loss, val_acc = do_eval(model, eval_dl, device)

        print(f"✅ Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
