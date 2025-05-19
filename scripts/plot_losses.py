import argparse
import matplotlib.pyplot as plt
import os
import csv

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("history_file", help="Path to training history TSV file")
args = parser.parse_args()

# Data containers
epochs = []
train_losses = []
val_losses = []
val_acc_avg = []
image_to_text_acc = []
text_to_image_acc = []

# Read TSV file
with open(args.history_file, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))
        val_acc_avg.append(float(row["val_acc_avg"]))
        image_to_text_acc.append(float(row["image_to_text_acc"]))
        text_to_image_acc.append(float(row["text_to_image_acc"]))

# Plotting
plt.figure(figsize=(12, 8))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Val Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(epochs, val_acc_avg, label="Val Acc Avg", marker='o')
plt.plot(epochs, image_to_text_acc, label="Image → Text Acc", marker='s')
plt.plot(epochs, text_to_image_acc, label="Text → Image Acc", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Metrics")
plt.legend()

plt.tight_layout()

# Save path setup
save_path = "/home/sysadm/Music/MedXpert/results/plots/loss_plot.png"
folder_path = os.path.dirname(save_path)

# ✅ Only save if folder already exists and file does not exist
if os.path.isdir(folder_path):
    if not os.path.exists(save_path):
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        print(f"⚠️ Plot not saved: File already exists at {save_path}")
else:
    print(f"❌ Folder '{folder_path}' does not exist. Plot not saved.")
