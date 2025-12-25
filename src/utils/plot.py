import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import Config


def plot_training_curves(history):
    """
    history = {
        "train_loss": [...],
        "val_loss": [...],
        "train_acc": [...],
        "val_acc": [...],
        "learning_rate": [...]
    }
    """
    save_dir = Config.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["learning_rate"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "learning_rate_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_labels=None):
    save_dir = Config.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    cm_indexed = cm.copy()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_indexed, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    num_classes = cm.shape[0]
    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks + 1)
    plt.yticks(ticks, ticks + 1)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
