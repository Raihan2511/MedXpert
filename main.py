# main.py
import os
import yaml
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datetime import timedelta
from transformers import get_scheduler
from models.clip.dataset import MIMICCLIPDataset
from models.clip.train import do_train, do_eval
from scripts.preprocess_data import preprocess_dataset

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
EMBED_DIR = config["embeddings"]["save_dir"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    print("Loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME)

    print("Preprocessing dataset...")
    preprocess_dataset(raw_dataset, output_dir="data/processed")

    train_ds = MIMICCLIPDataset(raw_dataset["train"], processor)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_ds = MIMICCLIPDataset(raw_dataset["validation"], processor)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

# Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    history_path = "results/train_history.tsv"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        train_loss,train_time = do_train(model, train_loader, optimizer, lr_scheduler, device)
        val_loss, val_acc ,val_time= do_eval(model, val_loader, device)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")

        with open(history_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
