# models/clip/train.py

# models/clip/train.py

import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader

import time
from datetime import timedelta

def do_train(model, train_dl, optimizer, lr_scheduler, device):
    train_loss = 0
    model.train()
    start_time = time.time()
    
    for bid, batch in enumerate(train_dl):
        batch_start = time.time()
        if bid % 100 == 0:
            print("...{:d} training steps complete".format(bid))

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss

        train_loss += loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if bid % 100 == 0 and bid > 0:
            batch_time = time.time() - batch_start
            eta = batch_time * (len(train_dl) - bid)
            print(f"    Batch time: {batch_time:.2f}s | ETA: {str(timedelta(seconds=int(eta)))}")

    total_time = time.time() - start_time
    avg_time_per_batch = total_time / len(train_dl)
    print(f"...{bid} training steps COMPLETE in {str(timedelta(seconds=int(total_time)))}")
    print(f"Average time per batch: {avg_time_per_batch:.2f}s")
    
    return train_loss, total_time

#     return val_loss, val_acc, total_time
def do_eval(model, eval_dl, device):
    model.eval()
    val_loss, val_acc, num_examples = 0, 0, 0
    start_time = time.time()
    
    # Add debugging to check if dataloader is empty
    print(f"Validation dataloader contains {len(eval_dl)} batches")
    
    for bid, batch in enumerate(eval_dl):
        # Print every batch during validation for debugging
        print(f"Validating batch {bid+1}/{len(eval_dl)}")
        
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_loss=True)

        loss = outputs.loss
        val_loss += loss.detach().cpu().numpy()

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predictions = torch.argmax(probs, dim=-1)
        labels = torch.arange(len(predictions)).to(device)

        accuracy = torch.sum(predictions == labels)
        num_examples += len(predictions)
        val_acc += accuracy

    total_time = time.time() - start_time
    
    # Avoid division by zero if no examples were processed
    if num_examples > 0:
        val_acc = val_acc.detach().cpu().numpy() / num_examples
    else:
        val_acc = 0.0
        print("WARNING: No examples were processed during validation!")
    
    print(f"Validation complete: Processed {num_examples} examples in {len(eval_dl)} batches")
    
    return val_loss, val_acc, total_time
