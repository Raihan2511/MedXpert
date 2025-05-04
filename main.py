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
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor and model
    print(f"Loading model: {MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    
    # Load and preprocess dataset
    print(f"Loading dataset: {DATASET_NAME}")
    raw_dataset = load_dataset(DATASET_NAME)
    
    print("Preprocessing dataset...")
    data_dir = "data/processed"
    preprocess_dataset(raw_dataset, output_dir="data/processed")
    print("Data preprocessing complete")
    print("\n*******************************************\n")
    print("Loading preprocessed dataset...")
    print(f"train: {len(raw_dataset['train'])}, validation: {len(raw_dataset['validation'])}, test: {len(raw_dataset['test'])}")
    print("\n*******************************************\n")
    print(raw_dataset["train"][0])
    print("\n*******************************************\n")
    # Create datasets and dataloaders
    print("Creating training datasets...")
    train_ds = MIMICCLIPDataset(data_dir, "train", processor)
    val_ds = MIMICCLIPDataset(data_dir, "validation", processor)
    test_ds = MIMICCLIPDataset(data_dir, "test", processor )
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print("\n*******************************************\n")
    print(train_ds[0])
    print("\n*******************************************\n")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

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
    os.makedirs(SAVE_PATH, exist_ok=True)  # <-- this line is key


    # for epoch in range(EPOCHS):
    #     print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
    #     train_loss,train_time = do_train(model, train_loader, optimizer, lr_scheduler, device)
    #     val_loss, val_acc ,val_time= do_eval(model, val_loader, device)
    #     print(f"Epoch {epoch+1} Summary:")
    #     print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
    #     print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")

    #     with open(history_path, "a") as f:
    #         f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")

    #     checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': train_loss,
    #         'val_loss': val_loss,
    #         'val_acc': val_acc,
    #     }, checkpoint_path)
    #     print(f"🧪 Checkpoint saved to {checkpoint_path}")



    # model.save_pretrained(SAVE_PATH)
    # processor.save_pretrained(SAVE_PATH)
    # print(f"✅ Final model saved to {SAVE_PATH}")

    best_val_loss = float("inf")
    patience = 3
    counter = 0

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_time = do_train(model, train_loader, optimizer, lr_scheduler, device)
        val_loss, val_acc, val_time = do_eval(model, val_loader, device)

        print("\n=================================================================================")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")        
        # Save training history
        with open(history_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
        
        checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f"🧪 Checkpoint saved to {checkpoint_path}")

        # 🛑 Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            print(f"⚠️ No improvement in val_loss. Patience counter: {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early stopping triggered!")
                break

    # ✅ Save final model
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"✅ Final model saved to {SAVE_PATH}")
    print("\n=================================================================================")


if __name__ == "__main__":
    main()
