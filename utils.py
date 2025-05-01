import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def val(model, val_loader, criterion, device):
    """
    Computes average validation loss for a regression task.
    """
    val_running_loss = 0.0
    total = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            total += 1

    return val_running_loss / total

def train(model, train_loader, val_loader, criterion, epochs, optimizer, device):
    train_loss_arr = []
    val_loss_arr = []
    
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch + 1}/{epochs}")

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val(model, val_loader, criterion, device)

        train_loss_arr.append(avg_train_loss)
        val_loss_arr.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    print("Training finished.")
    return train_loss_arr, val_loss_arr
