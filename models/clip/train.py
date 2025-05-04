# # models/clip/train.py

# # models/clip/train.py

# import torch
# from transformers import CLIPProcessor, CLIPModel
# from torch.utils.data import DataLoader

# import time
# from datetime import timedelta

# def do_train(model, train_dl, optimizer, lr_scheduler, device):
#     train_loss = 0
#     model.train()
#     start_time = time.time()
    
#     for bid, batch in enumerate(train_dl):
#         batch_start = time.time()
#         if bid % 100 == 0:
#             print("...{:d} training steps complete".format(bid))

#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch, return_loss=True)
#         loss = outputs.loss

#         train_loss += loss.detach().cpu().numpy()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
        
#         if bid % 100 == 0 and bid > 0:
#             batch_time = time.time() - batch_start
#             eta = batch_time * (len(train_dl) - bid)
#             print(f"    Batch time: {batch_time:.2f}s | ETA: {str(timedelta(seconds=int(eta)))}")

#     total_time = time.time() - start_time
#     avg_time_per_batch = total_time / len(train_dl)
#     print(f"...{bid} training steps COMPLETE in {str(timedelta(seconds=int(total_time)))}")
#     print(f"Average time per batch: {avg_time_per_batch:.2f}s")
    
#     return train_loss, total_time

# #     return val_loss, val_acc, total_time
# def do_eval(model, eval_dl, device):
#     model.eval()
#     val_loss, val_acc, num_examples = 0, 0, 0
#     start_time = time.time()
    
#     # Add debugging to check if dataloader is empty
#     print(f"Validation dataloader contains {len(eval_dl)} batches")
    
#     for bid, batch in enumerate(eval_dl):
#         # Print every batch during validation for debugging
#         print(f"Validating batch {bid+1}/{len(eval_dl)}")
        
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch, return_loss=True)

#         loss = outputs.loss
#         val_loss += loss.detach().cpu().numpy()

#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1)
#         predictions = torch.argmax(probs, dim=-1)
#         labels = torch.arange(len(predictions)).to(device)

#         accuracy = torch.sum(predictions == labels)
#         num_examples += len(predictions)
#         val_acc += accuracy

#     total_time = time.time() - start_time
    
#     # Avoid division by zero if no examples were processed
#     if num_examples > 0:
#         val_acc = val_acc.detach().cpu().numpy() / num_examples
#     else:
#         val_acc = 0.0
#         print("WARNING: No examples were processed during validation!")
    
#     print(f"Validation complete: Processed {num_examples} examples in {len(eval_dl)} batches")
    
#     return val_loss, val_acc, total_time

import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import numpy as np

def do_train(model, train_dl, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    total_batches = 0
    start_time = time.time()
    
    # Track metrics for reporting
    running_loss = 0
    log_interval = 100
    
    for bid, batch in enumerate(train_dl):
        batch_start = time.time()
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Accumulate loss (use float value to avoid GPU memory buildup)
        batch_loss = loss.item()
        total_loss += batch_loss
        total_batches += 1
        running_loss += batch_loss
        
        # Logging
        if bid % log_interval == 0:
            if bid > 0:
                avg_running_loss = running_loss / log_interval
                running_loss = 0
                batch_time = time.time() - batch_start
                eta = batch_time * (len(train_dl) - bid)
                
                print(f"Batch {bid}/{len(train_dl)} | " 
                      f"Loss: {avg_running_loss:.4f} | "
                      f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | "
                      f"ETA: {str(timedelta(seconds=int(eta)))}")
            else:
                print(f"Starting training on {len(train_dl)} batches...")
    
    # Calculate average loss over all batches
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    total_time = time.time() - start_time
    
    print(f"Training complete: {total_batches} batches in {str(timedelta(seconds=int(total_time)))}")
    print(f"Average batch time: {total_time/total_batches:.2f}s | Average loss: {avg_loss:.4f}")
    
    return avg_loss, total_time


def do_eval(model, eval_dl, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_examples = 0
    start_time = time.time()
    
    # Make sure we have data to validate on
    if len(eval_dl) == 0:
        print("Warning: Empty validation dataloader!")
        return 0, 0, 0
    
    print(f"Starting validation on {len(eval_dl)} batches...")
    
    for bid, batch in enumerate(eval_dl):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(**batch, return_loss=True)
        
        # Calculate loss and accumulate
        loss = outputs.loss
        total_loss += loss.item()
        
        # Calculate accuracy: in CLIP, the diagonal elements should have the highest score
        logits_per_image = outputs.logits_per_image
        batch_size = logits_per_image.size(0)
        
        if batch_size == 0:
            print(f"Warning: Batch {bid} has size 0")
            continue
            
        # The expected labels in CLIP are the diagonal (matching text-image pairs)
        labels = torch.arange(batch_size).to(device)
        predictions = torch.argmax(logits_per_image, dim=1)
        
        # Count correct predictions
        batch_correct = (predictions == labels).sum().item()
        correct += batch_correct
        total_examples += batch_size
        
        # Log progress
        if (bid + 1) % 10 == 0 or (bid + 1) == len(eval_dl):
            print(f"Validated {bid+1}/{len(eval_dl)} batches | "
                  f"Current Accuracy: {100 * correct / total_examples:.2f}%")
    
    # Calculate averages
    avg_loss = total_loss / len(eval_dl) if len(eval_dl) > 0 else 0
    accuracy = correct / total_examples if total_examples > 0 else 0
    total_time = time.time() - start_time
    
    print(f"Validation complete: {total_examples} examples in {str(timedelta(seconds=int(total_time)))}")
    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {100 * accuracy:.2f}%")
    
    return avg_loss, accuracy, total_time