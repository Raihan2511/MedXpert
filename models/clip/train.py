# models/clip/train.py

import torch

def do_train(model, train_dl, optimizer, lr_scheduler, device):
    train_loss = 0
    model.train()
    for bid, (batch, _) in enumerate(train_dl):
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

    print("...{:d} training steps COMPLETE".format(bid))
    return train_loss


def do_eval(model, eval_dl, device):
    model.eval()
    val_loss, val_acc, num_examples = 0, 0, 0

    for bid, (batch, _) in enumerate(eval_dl):
        if bid % 100 == 0:
            print("... {:d} validation steps complete".format(bid))

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

    print("... {:d} validation steps COMPLETE".format(bid))
    val_acc = val_acc.detach().cpu().numpy() / num_examples
    return val_loss, val_acc