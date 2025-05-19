# CLAUDE SUGGESTION:

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
from tqdm import tqdm

# def do_train(model, train_loader, optimizer, epoch, device, scheduler=None, max_grad_norm=1.0):
#     model.train()
#     total_loss = 0.0
#     start_time = time.time()
    
#     for batch_idx, batch in enumerate(train_loader):
#         batch = {k: v.to(device) for k, v in batch.items()}
        
#         # Forward pass
#         outputs = model(**batch, return_loss=True)
#         loss = outputs.loss
        
#         # Backward pass
#         loss.backward()
        
#         # Gradient clipping for stability
#         if max_grad_norm > 0:
#             clip_grad_norm_(model.parameters(), max_grad_norm)
            
#         optimizer.step()
#         optimizer.zero_grad()
        
#         total_loss += loss.item()
        
#         # Report progress for long epochs
#         if batch_idx % 50 == 0:
#             print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
#     # Step the scheduler if provided
#     if scheduler is not None:
#         scheduler.step()
    
#     train_time = time.time() - start_time
#     avg_train_loss = total_loss / len(train_loader)
    
#     return avg_train_loss, train_time

# @torch.no_grad()
# def do_eval(model, val_loader, device):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
#     start_time = time.time()
    
#     for batch in val_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
        
#         # Get loss
#         outputs = model(**batch, return_loss=True)
#         loss = outputs.loss
#         total_loss += loss.item()
        
#         # Get image and text features
#         image_embeds = model.get_image_features(pixel_values=batch['pixel_values'])
#         text_embeds = model.get_text_features(
#             input_ids=batch['input_ids'], 
#             attention_mask=batch['attention_mask']
#         )
        
#         # Normalize
#         image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
#         text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
#         # Compute similarity
#         logits = torch.matmul(image_embeds, text_embeds.T) * model.logit_scale.exp()
        
#         # Prediction (diagonal elements should be highest)
#         preds = torch.argmax(logits, dim=1)
#         labels = torch.arange(len(preds)).to(device)
        
#         all_preds.extend(preds.cpu().tolist())
#         all_labels.extend(labels.cpu().tolist())
    
#     val_time = time.time() - start_time
#     avg_val_loss = total_loss / len(val_loader)
#     val_acc = accuracy_score(all_labels, all_preds)
    
    # return avg_val_loss, val_acc, val_time

# CLAUDE SUGGESTION:

# CLAUDE SUGGESTION:



# def do_train(model, train_loader, optimizer, epoch, device, scheduler=None, max_grad_norm=1.0):
#     model.train()
#     total_loss = 0.0
#     start_time = time.time()
    
#     for batch_idx, batch in enumerate(train_loader):
#         batch = {k: v.to(device) for k, v in batch.items()}
        
#         # Forward pass
#         outputs = model(**batch, return_loss=True)
#         loss = outputs.loss
        
#         # Backward pass
#         loss.backward()
        
#         # Gradient clipping for stability
#         if max_grad_norm > 0:
#             clip_grad_norm_(model.parameters(), max_grad_norm)
            
#         optimizer.step()
#         optimizer.zero_grad()
        
#         total_loss += loss.item()
        
#         # Report progress for long epochs
#         if batch_idx % 50 == 0:
#             print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
#     # Step the scheduler if provided
#     if scheduler is not None:
#         scheduler.step()
    
#     train_time = time.time() - start_time
#     avg_train_loss = total_loss / len(train_loader)
    
#     return avg_train_loss, train_time

# @torch.no_grad()
# def do_eval(model, val_loader, device):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
#     start_time = time.time()
    
#     for batch in val_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
        
#         # Get loss
#         outputs = model(**batch, return_loss=True)
#         loss = outputs.loss
#         total_loss += loss.item()
        
#         # Get image and text features
#         image_embeds = model.get_image_features(pixel_values=batch['pixel_values'])
#         text_embeds = model.get_text_features(
#             input_ids=batch['input_ids'], 
#             attention_mask=batch['attention_mask']
#         )
        
#         # Normalize
#         image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
#         text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
#         # Compute similarity
#         logits = torch.matmul(image_embeds, text_embeds.T) * model.logit_scale.exp()
        
#         # Prediction (diagonal elements should be highest)
#         preds = torch.argmax(logits, dim=1)
#         labels = torch.arange(len(preds)).to(device)
        
#         all_preds.extend(preds.cpu().tolist())
#         all_labels.extend(labels.cpu().tolist())
    
#     val_time = time.time() - start_time
#     avg_val_loss = total_loss / len(val_loader)
#     val_acc = accuracy_score(all_labels, all_preds)
    
#     return avg_val_loss, val_acc, val_time

# def do_train(model, train_loader, optimizer, epoch, device, scheduler=None, max_grad_norm=1.0, log_interval=10):
#     model.train()
#     total_loss = 0.0
#     start_time = time.time()
    
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
#     for batch_idx, batch in enumerate(progress_bar):
#         pixel_values = batch["pixel_values"].to(device)
#         input_ids = batch["input_ids"].to(device)

#         batch_size = pixel_values.size(0)
#         if batch_size < 2:
#             continue  # Skip too-small batches

#         # Encode
#         image_features = model.get_image_features(pixel_values=pixel_values)
#         text_features = model.get_text_features(input_ids=input_ids)


#         # Normalize
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # Compute logits
#         logit_scale = model.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.T
#         logits_per_text = logits_per_image.T

#         # Compute loss
#         labels = torch.arange(batch_size, device=device)
#         loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

#         # Backprop
#         optimizer.zero_grad()
#         loss.backward()

#         if max_grad_norm > 0:
#             clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()

#         total_loss += loss.item()
#         progress_bar.set_postfix({"loss": loss.item()})

#         if batch_idx % log_interval == 0:
#             lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}, LR: {lr:.6f}")

#     if scheduler is not None:
#         scheduler.step()

#     train_time = time.time() - start_time
#     avg_train_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch}: Avg Loss: {avg_train_loss:.4f}, Time: {train_time:.2f}s")

#     return avg_train_loss, train_time
# @torch.no_grad()
# def do_eval(model, val_loader, device):
#     model.eval()
#     total_loss = 0.0
#     all_image_to_text_preds, all_image_to_text_labels = [], []
#     all_text_to_image_preds, all_text_to_image_labels = [], []
#     all_logits_per_image, all_logits_per_text = [], []
#     start_time = time.time()
    
#     progress_bar = tqdm(val_loader, desc="Evaluation")

#     for batch in progress_bar:
#         pixel_values = batch["pixel_values"].to(device)
#         input_ids = batch["input_ids"].to(device)

#         batch_size = pixel_values.size(0)
#         if batch_size < 2:
#             continue  # Skip batches too small for contrastive loss

#         # Encode
#         image_features = model.get_image_features(pixel_values=pixel_values)
#         text_features = model.get_text_features(input_ids=input_ids)


#         # Normalize
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # Compute logits
#         logit_scale = model.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.T
#         logits_per_text = logits_per_image.T

#         labels = torch.arange(batch_size, device=device)
#         loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
#         total_loss += loss.item()

#         image_to_text_preds = torch.argmax(logits_per_image, dim=1)
#         text_to_image_preds = torch.argmax(logits_per_text, dim=1)

#         all_image_to_text_preds.extend(image_to_text_preds.cpu().tolist())
#         all_image_to_text_labels.extend(labels.cpu().tolist())
#         all_text_to_image_preds.extend(text_to_image_preds.cpu().tolist())
#         all_text_to_image_labels.extend(labels.cpu().tolist())
#         all_logits_per_image.append(logits_per_image.cpu())
#         all_logits_per_text.append(logits_per_text.cpu())

#     val_time = time.time() - start_time
#     avg_val_loss = total_loss / len(val_loader)

#     image_to_text_acc = accuracy_score(all_image_to_text_labels, all_image_to_text_preds)
#     text_to_image_acc = accuracy_score(all_text_to_image_labels, all_text_to_image_preds)

#     # Top-5
#     top5_image_scores, top5_text_scores = [], []
#     for logits_img, logits_txt in zip(all_logits_per_image, all_logits_per_text):
#         batch_size = logits_img.shape[0]
#         if batch_size < 2:
#             continue
#         labels = list(range(batch_size))
#         k = min(5, batch_size)
#         top5_image_scores.append(top_k_accuracy_score(labels, logits_img.numpy(), k=k, labels=labels))
#         top5_text_scores.append(top_k_accuracy_score(labels, logits_txt.numpy(), k=k, labels=labels))

#     image_to_text_top5 = sum(top5_image_scores) / len(top5_image_scores) if top5_image_scores else 0
#     text_to_image_top5 = sum(top5_text_scores) / len(top5_text_scores) if top5_text_scores else 0

#     print(f"Validation Results:")
#     print(f"  Loss: {avg_val_loss:.4f}")
#     print(f"  Image→Text Accuracy: {image_to_text_acc:.4f}")
#     print(f"  Text→Image Accuracy: {text_to_image_acc:.4f}")
#     print(f"  Image→Text Top-5: {image_to_text_top5:.4f}")
#     print(f"  Text→Image Top-5: {text_to_image_top5:.4f}")

#     return {
#         "avg_val_loss": avg_val_loss,
#         "image_to_text_acc": image_to_text_acc,
#         "text_to_image_acc": text_to_image_acc,
#         "image_to_text_top5": image_to_text_top5,
#         "text_to_image_top5": text_to_image_top5,
#         "val_time": val_time
#     }

def do_train(model, train_loader, optimizer, epoch, device, scheduler=None, max_grad_norm=1.0, log_interval=10):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get image and text features
        image_features = model.get_image_features(pixel_values=batch['pixel_values'])
        text_features = model.get_text_features(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None)  # Handle if attention_mask isn't available
        )
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # Compute loss (bidirectional)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=device)
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
        if batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}, LR: {lr:.6f}")
    
    # Step the scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    train_time = time.time() - start_time
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Avg Loss: {avg_train_loss:.4f}, Time: {train_time:.2f}s")
    
    return avg_train_loss, train_time
@torch.no_grad()
def do_eval(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_image_to_text_preds, all_image_to_text_labels = [], []
    all_text_to_image_preds, all_text_to_image_labels = [], []
    all_logits_per_image, all_logits_per_text = [], []
    start_time = time.time()
    
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get loss
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Get image and text features
        image_embeds = model.get_image_features(pixel_values=batch['pixel_values'])
        text_embeds = model.get_text_features(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None)  # Handle if attention_mask isn't available
        )
        
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarities in both directions
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_embeds, text_embeds.T)
        logits_per_text = logits_per_image.T
        
        # Create labels (diagonal elements should match)
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size).to(device)
        
        # Get predictions for both directions
        image_to_text_preds = torch.argmax(logits_per_image, dim=1)
        text_to_image_preds = torch.argmax(logits_per_text, dim=1)
        
        # Store results for later computation
        all_image_to_text_preds.extend(image_to_text_preds.cpu().tolist())
        all_image_to_text_labels.extend(labels.cpu().tolist())
        all_text_to_image_preds.extend(text_to_image_preds.cpu().tolist())
        all_text_to_image_labels.extend(labels.cpu().tolist())
        
        # Store logits for computing top-5 metrics
        all_logits_per_image.append(logits_per_image.cpu())
        all_logits_per_text.append(logits_per_text.cpu())
    
    # Compute metrics
    val_time = time.time() - start_time
    avg_val_loss = total_loss / len(val_loader)
    
    # Compute regular accuracy metrics
    image_to_text_acc = accuracy_score(all_image_to_text_labels, all_image_to_text_preds)
    text_to_image_acc = accuracy_score(all_text_to_image_labels, all_text_to_image_preds)
    
    # Compute top-5 metrics
    top5_image_scores, top5_text_scores = [], []
    for logits_img, logits_txt in zip(all_logits_per_image, all_logits_per_text):
        batch_size = logits_img.shape[0]
        labels = list(range(batch_size))
        k = min(5, batch_size)  # Handle batches smaller than 5
        
        # Use sklearn's top_k_accuracy_score
        top5_image_scores.append(top_k_accuracy_score(labels, logits_img.numpy(), k=k, labels=labels))
        top5_text_scores.append(top_k_accuracy_score(labels, logits_txt.numpy(), k=k, labels=labels))
    
    # Average top-5 scores across batches
    image_to_text_top5 = sum(top5_image_scores) / len(top5_image_scores) if top5_image_scores else 0
    text_to_image_top5 = sum(top5_text_scores) / len(top5_text_scores) if top5_text_scores else 0
    
    # Display results
    print(f"Validation Results:")
    print(f"  Loss: {avg_val_loss:.4f}")
    print(f"  Image→Text Accuracy: {image_to_text_acc:.4f} (Top-5: {image_to_text_top5:.4f})")
    print(f"  Text→Image Accuracy: {text_to_image_acc:.4f} (Top-5: {text_to_image_top5:.4f})")
    
    # Return all metrics
    return {
        "avg_val_loss": avg_val_loss,
        "image_to_text_acc": image_to_text_acc,
        "text_to_image_acc": text_to_image_acc,
        "image_to_text_top5": image_to_text_top5,
        "text_to_image_top5": text_to_image_top5,
        "val_time": val_time
    }