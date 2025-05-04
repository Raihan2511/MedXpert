import argparse
import matplotlib.pyplot as plt
import os

# Argument parser for history file
parser = argparse.ArgumentParser()
parser.add_argument("history_file", help="Path to training history TSV file")
args = parser.parse_args()

# Lists to store parsed values
epochs, train_losses, val_losses, val_accs = [], [], [], []

# Parse TSV history file
with open(args.history_file, "r") as fhist:
    for line in fhist:
        epoch, train_loss, val_loss, val_acc = line.strip().split('\t')
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='x')
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.subplot(2, 1, 2)
plt.plot(epochs, val_accs, label="Validation Accuracy", color='green', marker='s')
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")

plt.tight_layout()

# Save to file
save_path = "results/plots/loss_plot.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
print(f"✅ Plot saved to {save_path}")
