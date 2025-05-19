# # main.py
# import os
# import yaml
# import torch
# from transformers import CLIPModel, CLIPProcessor
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from datetime import timedelta
# from transformers import get_scheduler
# from models.clip.dataset import MIMICCLIPDataset
# from models.clip.train import do_train, do_eval
# from scripts.preprocess_data import preprocess_dataset
# from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau

# # Load config
# with open("config/clip_config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# MODEL_NAME = config["model"]["name"]
# DATASET_NAME = config["dataset"]["name"]
# USE_FINDINGS = config["dataset"].get("use_findings_if_missing", True)
# BATCH_SIZE = config["training"]["batch_size"]
# EPOCHS = int(config["training"]["epochs"])
# LR = float(config["training"]["learning_rate"])
# SAVE_PATH = config["training"]["save_path"]
# EMBED_DIR = config["embeddings"]["save_dir"]


# def main():
#     # Setup device
#     # print(LR.type())
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Load processor and model
#     print(f"Loading model: {MODEL_NAME}")
#     processor = CLIPProcessor.from_pretrained(MODEL_NAME)
#     model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    
#     # Load and preprocess dataset
#     print(f"Loading dataset: {DATASET_NAME}")
#     raw_dataset = load_dataset(DATASET_NAME)
    
#     print("Preprocessing dataset...")
#     data_dir = "data/processed"
#     preprocess_dataset(raw_dataset, output_dir="data/processed")
#     print("Data preprocessing complete")
#     print("\n*******************************************\n")
#     print("Loading preprocessed dataset...")
#     print(f"train: {len(raw_dataset['train'])}, validation: {len(raw_dataset['validation'])}, test: {len(raw_dataset['test'])}")
#     print("\n*******************************************\n")
#     print(raw_dataset["train"][0])
#     print("\n*******************************************\n")
#     # Create datasets and dataloaders
#     print("Creating training datasets...")
#     train_ds = MIMICCLIPDataset(data_dir, "train", processor)
#     val_ds = MIMICCLIPDataset(data_dir, "validation", processor)
#     test_ds = MIMICCLIPDataset(data_dir, "test", processor )
    
#     print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
#     print("\n*******************************************\n")
#     print(train_ds[0])
#     print("\n*******************************************\n")
    
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#     num_epochs=EPOCHS,
#     learning_rate=LR
#     weight_decay=0.01,
#     scheduler_type="cosine",  # Options: "cosine", "linear", "plateau", "none"
#     warmup_epochs=1,

#     # Setup optimizer and scheduler
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#     num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
#     lr_scheduler = get_linear_schedule_with_warmup(
#         optimizer, 
#         num_warmup_steps=0,
#         num_training_steps=num_training_steps
#     )
#         # Create optimizer - AdamW is typically used for transformers
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay
#     )
    
#     # Create scheduler based on type
#     scheduler = None
#     if scheduler_type == "cosine":
#         # Cosine decay from initial lr to 0
#         scheduler = CosineAnnealingLR(
#             optimizer, 
#             T_max=num_epochs - warmup_epochs,
#             eta_min=1e-6
#         )
        
#         # Optional: Warmup scheduler for first few epochs
#         if warmup_epochs > 0:
#             warmup_scheduler = LinearLR(
#                 optimizer, 
#                 start_factor=0.1, 
#                 end_factor=1.0, 
#                 total_iters=warmup_epochs
#             )
            
#     elif scheduler_type == "linear":
#         # Linear decay from initial lr to 0
#         scheduler = LinearLR(
#             optimizer,
#             start_factor=1.0,
#             end_factor=0.1,
#             total_iters=num_epochs
#         )
        
#     elif scheduler_type == "plateau":
#         # Reduce LR when validation loss plateaus
#         scheduler = ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=0.5,
#             patience=2,
#             verbose=True
#         )

#     history_path = "results/train_history.tsv"
#     os.makedirs(os.path.dirname(history_path), exist_ok=True)
#     os.makedirs(SAVE_PATH, exist_ok=True)  # <-- this line is key


#     # for epoch in range(EPOCHS):
#     #     print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
#     #     train_loss,train_time = do_train(model, train_loader, optimizer, lr_scheduler, device)
#     #     val_loss, val_acc ,val_time= do_eval(model, val_loader, device)
#     #     print(f"Epoch {epoch+1} Summary:")
#     #     print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
#     #     print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")

#     #     with open(history_path, "a") as f:
#     #         f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")

#     #     checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
#     #     torch.save({
#     #         'epoch': epoch,
#     #         'model_state_dict': model.state_dict(),
#     #         'optimizer_state_dict': optimizer.state_dict(),
#     #         'train_loss': train_loss,
#     #         'val_loss': val_loss,
#     #         'val_acc': val_acc,
#     #     }, checkpoint_path)
#     #     print(f"🧪 Checkpoint saved to {checkpoint_path}")



#     # model.save_pretrained(SAVE_PATH)
#     # processor.save_pretrained(SAVE_PATH)
#     # print(f"✅ Final model saved to {SAVE_PATH}")

#     best_val_loss = float("inf")
#     patience = 3
#     counter = 0

#     for epoch in range(EPOCHS):
#         print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
#         # train_loss, train_time = do_train(model, train_loader, optimizer, lr_scheduler, device)
#         train_loss, train_time = do_train(
#         model,
#         train_loader,
#         optimizer,
#         epoch,
#         device,
#         scheduler=(warmup_scheduler if epoch < warmup_epochs and warmup_epochs > 0 else None),
#         max_grad_norm=1.0
#         )
#         val_loss, val_acc, val_time = do_eval(model, val_loader, device)
#         # Step the scheduler
#         if scheduler_type == "plateau":
#             scheduler.step(val_loss)
#         elif scheduler is not None and epoch >= warmup_epochs:
#             scheduler.step()
#         elif warmup_epochs > 0 and epoch < warmup_epochs:
#             warmup_scheduler.step()

#         print("\n=================================================================================")
#         print(f"Epoch {epoch+1} Summary:")
#         print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
#         print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")        
#         # Save training history
#         with open(history_path, "a") as f:
#             f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
        
#         checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'val_acc': val_acc,
#         }, checkpoint_path)
#         print(f"🧪 Checkpoint saved to {checkpoint_path}")

#         # 🛑 Early Stopping logic
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             counter = 0
#         else:
#             counter += 1
#             print(f"⚠️ No improvement in val_loss. Patience counter: {counter}/{patience}")
#             if counter >= patience:
#                 print("🛑 Early stopping triggered!")
#                 break

#     # ✅ Save final model
#     model.save_pretrained(SAVE_PATH)
#     processor.save_pretrained(SAVE_PATH)
#     print(f"✅ Final model saved to {SAVE_PATH}")
#     print("\n=================================================================================")


# if __name__ == "__main__":
#     main()


# # Define the CLIPDataset class
# class CLIPDataset(torch.utils.data.Dataset):
#     def __init__(self, hf_dataset):
#         self.dataset = hf_dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]
#         image = example["image"].convert("RGB")
#         text = example["impression"]
#         return {"image": image, "text": text}

# Define collate function
# def collate_fn(batch, processor):
#     images = [b["image"] for b in batch]
    
#     # Handle text inputs - ensure they are strings and handle potential None values
#     texts = []
#     for b in batch:
#         text = b["text"]
#         if text is None:
#             text = ""  # Replace None with empty string
#         elif not isinstance(text, str):
#             text = str(text)  # Convert non-string to string
#         texts.append(text)
    
#     # Process images and text separately
#     # Use image_processor instead of feature_extractor to avoid deprecation warning
#     inputs = processor.image_processor(images=images, return_tensors="pt")
#     inputs.update(processor.tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt"))

    
# def clip_collate_fn(batch, processor):
#     images = [item["image"] for item in batch]
#     texts = [item["text"] for item in batch]
#     image_ids = [item["image_id"] for item in batch]
    
#     # Process images - transformation is already applied in the dataset class
#     if isinstance(images[0], torch.Tensor):
#         images = torch.stack(images)
    
#     # Process text using CLIP's tokenizer from the processor
#     tokenized = processor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    
#     return {
#         "pixel_values": images,
#         "input_ids": tokenized,
#         "texts": texts,  # Keep original texts for debugging
#         "image_ids": image_ids  # Keep IDs for evaluation
#     }

# def clip_collate_fn(batch, processor):
#     images = [item["image"] for item in batch]
#     texts = [item["text"] for item in batch]
#     image_ids = [item["image_id"] for item in batch]

#     # Use processor to handle both image & text
#     processed = processor(
#         text=texts,
#         images=images,
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     )

#     return {
#         "pixel_values": processed["pixel_values"],
#         "input_ids": processed["input_ids"],
#         "attention_mask": processed["attention_mask"],
#         "texts": texts,
#         "image_ids": image_ids
#     }

    # Set up transforms for training and validation
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
    #                         (0.26862954, 0.26130258, 0.27577711))
    # ])
    
    # val_transforms = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
    #                         (0.26862954, 0.26130258, 0.27577711))
    # ])
    
    # Create datasets with appropriate transforms
    # print("Creating training datasets...")
    # train_ds = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/train.json", 
    #     root_dir="data/processed",
    #     # transform=train_transforms  # Apply training transforms
    # )
    
    # val_ds = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/validation.json", 
    #     root_dir="data/processed",
    #     # transform=val_transforms  # Apply validation transforms
    # )
    
    # test_ds = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/test.json", 
    #     root_dir="data/processed",
    #     # transform=val_transforms  # Use validation transforms for test set too
    # )


import os
# Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import timedelta
from torch.nn.utils import clip_grad_norm_
from models.clip.dataset import CLIPDataset
from models.clip.dataset import ProcessedCLIPDataset
from transformers import get_linear_schedule_with_warmup
from transformers import get_scheduler
from scripts.preprocess_data import preprocess_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
from models.clip.train import do_train, do_eval
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms
from scripts.save_embedding import save_all_embeddings


def clip_collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return inputs

def main():
    with open("config/clip_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    MODEL_NAME = config["model"]["name"]
    DATASET_NAME = config["dataset"]["name"]
    USE_FINDINGS = config["dataset"].get("use_findings_if_missing", True)
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = int(config["training"]["epochs"])
    LR = float(config["training"]["learning_rate"])
    SAVE_PATH = config["training"]["save_path"]
    EMBED_DIR = config["embeddings"]["save_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    if "validation" not in dataset or "test" not in dataset:
        print("\n🔀 Splitting dataset into 85% train / 10% val / 5% test...")
        dataset["train"] = dataset["train"].shuffle(seed=42)
        total = len(dataset["train"])
        train_end = int(0.85 * total)
        val_end = train_end + int(0.10 * total)

        dataset["validation"] = dataset["train"].select(range(train_end, val_end))
        dataset["test"] = dataset["train"].select(range(val_end, total))
        dataset["train"] = dataset["train"].select(range(0, train_end))

    total = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
    print("\n##############################################################################################")
    print(f"✅ Dataset split summary:")
    print(f"  Train: {len(dataset['train'])} samples ({len(dataset['train']) / total * 100:.2f}%)")
    print(f"  Val:   {len(dataset['validation'])} samples ({len(dataset['validation']) / total * 100:.2f}%)")
    print(f"  Test:  {len(dataset['test'])} samples ({len(dataset['test']) / total * 100:.2f}%)")
    print("\n##############################################################################################")
    # print("Preprocessing dataset...")
    # data_dir = "data/processed"
    # preprocess_dataset(dataset, output_dir="data/processed")
    # print("Data preprocessing complete")
    # print("\n*******************************************\n")
    # print("Loading preprocessed dataset...")

    # print("\n##############################################################################################")
    # print(f"✅ Dataset split summary:")
    # print(f"  Train: {len(dataset['train'])} samples ({len(dataset['train']) / total * 100:.2f}%)")
    # print(f"  Val:   {len(dataset['validation'])} samples ({len(dataset['validation']) / total * 100:.2f}%)")
    # print(f"  Test:  {len(dataset['test'])} samples ({len(dataset['test']) / total * 100:.2f}%)")
    # print("\n##############################################################################################")

    train_dataset = CLIPDataset(dataset["train"])
    val_dataset = CLIPDataset(dataset["validation"])
    test_dataset = CLIPDataset(dataset["test"])

    # train_dataset = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/train.json", 
    #     root_dir="data/processed",
    #     # transform=train_transforms  # Apply training transforms
    # )
    # val_dataset = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/validation.json", 
    #     root_dir="data/processed",
    # )
    # test_dataset = ProcessedCLIPDataset(
    #     json_path="data/processed/texts/test.json", 
    #     root_dir="data/processed",
    # )
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print("\n*******************************************\n")
    custom_collate = lambda batch: clip_collate_fn(batch, processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=LR)

    history_path = "results/train_history.tsv"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)

    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            f.write("epoch\ttrain_loss\tval_loss\tval_acc_avg\timage_to_text_acc\timage_to_text_top5\ttext_to_image_acc\ttext_to_image_top5\n")

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_time = do_train(model, train_loader, optimizer, epoch, device=device)
        eval_results = do_eval(model, val_loader, device=device)

        val_loss = eval_results["avg_val_loss"]
        image_to_text_acc = eval_results["image_to_text_acc"]
        text_to_image_acc = eval_results["text_to_image_acc"]
        image_to_text_top5 = eval_results["image_to_text_top5"]
        text_to_image_top5 = eval_results["text_to_image_top5"]
        val_time = eval_results["val_time"]

        val_acc_avg = (image_to_text_acc + text_to_image_acc) / 2

        print("\n=================================================================================")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Time: {str(timedelta(seconds=int(train_time)))}")
        print(f"  Val Loss: {val_loss:.4f} | Val Time: {str(timedelta(seconds=int(val_time)))}")
        print(f"  Image→Text: {image_to_text_acc:.4f} (Top-5: {image_to_text_top5:.4f})")
        print(f"  Text→Image: {text_to_image_acc:.4f} (Top-5: {text_to_image_top5:.4f})")

        with open(history_path, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc_avg:.4f}\t"
                    f"{image_to_text_acc:.4f}\t{image_to_text_top5:.4f}\t{text_to_image_acc:.4f}\t{text_to_image_top5:.4f}\n")

        checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'image_to_text_acc': image_to_text_acc,
            'text_to_image_acc': text_to_image_acc,
        }, checkpoint_path)
        print(f"🧪 Checkpoint saved to {checkpoint_path}")

        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            best_epoch = epoch + 1

            best_model_path = os.path.join(SAVE_PATH, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'image_to_text_acc': image_to_text_acc,
                'text_to_image_acc': text_to_image_acc,
                'val_acc_avg': val_acc_avg,
            }, best_model_path)
            print(f"🏆 New best model saved! Validation Accuracy: {val_acc_avg:.4f}")

    print(f"\n🥇 Best model was from epoch {best_epoch} with validation accuracy: {best_val_acc:.4f}")

    print("\n=================================================================================")
    print("🔍 Final Evaluation on Test Set")
    test_results = do_eval(model, test_loader, device=device)

    test_loss = test_results["avg_val_loss"]
    test_i2t_acc = test_results["image_to_text_acc"]
    test_t2i_acc = test_results["text_to_image_acc"]
    test_i2t_top5 = test_results["image_to_text_top5"]
    test_t2i_top5 = test_results["text_to_image_top5"]
    test_time = test_results["val_time"]

    print(f"📊 Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Image→Text: {test_i2t_acc:.4f} (Top-5: {test_i2t_top5:.4f})")
    print(f"  Text→Image: {test_t2i_acc:.4f} (Top-5: {test_t2i_top5:.4f})")

    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"✅ Final model saved to {SAVE_PATH}")

    print("\n💾 Saving all embeddings...")
    save_all_embeddings(model, processor, device, dataset)
    print("✅ Embeddings saved for train/val/test")

    print("\n=================================================================================")

if __name__ == "__main__":
    main()
