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

        tqdm.write(f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in tqdm(train_loader, desc="train"):
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

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

    print("Training finished.")
    return train_loss_arr, val_loss_arr


def val_self_supervised(model, val_loader, self_supervised_loss, device):
    val_running_loss = 0.0
    total = 0

    model.eval()
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device).float()

            reconstructed, original, mask = model(inputs)
            loss = self_supervised_loss(reconstructed, original, mask)

            val_running_loss += loss.item()
            total += 1

    return val_running_loss / total


def train_self_supervised(
    model, train_loader, val_loader, self_supervised_loss, epochs, optimizer, device
):
    train_loss_arr = []
    val_loss_arr = []
    print("Starting training...")
    for epoch in range(epochs):
        tqdm.write(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for x in tqdm(train_loader, desc="train"):
            x = x.to(device)
            reconstructed, original, mask = model(x)
            loss = self_supervised_loss(reconstructed, original, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_self_supervised(
            model, val_loader, self_supervised_loss, device
        )
        train_loss_arr.append(avg_loss)
        val_loss_arr.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

    print("Training finished.")
    return train_loss_arr, val_loss_arr
