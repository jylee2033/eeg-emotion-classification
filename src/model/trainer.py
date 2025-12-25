import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import Config
from utils.plot import plot_training_curves, plot_confusion_matrix


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    device = Config.DEVICE
    best_val_loss = float("inf")

    # For plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rate": [],
    }

    print("\nStep 3: Starting CNN Training...\n")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        correct = 0
        total = 0
        train_loss = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        # validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_truth = []

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                _, preds = outputs.max(1)
                val_total += y.size(0)
                val_correct += preds.eq(y).sum().item()

                val_preds.extend(preds.cpu().numpy())
                val_truth.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Improved val loss → {val_loss:.4f}. Saving model.")

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
    
    # plot
    print("\nGenerating training plots...")
    plot_training_curves(history)
    print("Training plots saved.")

    print("Creating final confusion matrix...")
    plot_confusion_matrix(val_truth, val_preds)
    print("Confusion matrix saved.")
