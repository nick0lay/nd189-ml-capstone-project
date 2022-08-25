import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

use_scheduler = True

def train_model(model, loader_train, loader_valid, epochs=30, lr=0.0001):
    loss_fn = nn.MSELoss(reduction='mean') # default 'mean'
    # loss_fn = nn.L1Loss();
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # training always has a shift from orifgnal series
    if use_scheduler:
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8)

    train_losses = []
    valid_losses = []
    last_valid_loss = None
    min_epochs = 1000 # use 1000 to disable early stopping
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, loader_train, loss_fn, optimizer)
        valid_loss = validate(model, loader_valid, loss_fn)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # early training stopping
        if min_epochs <= epoch and last_valid_loss < valid_loss:
            # exit model starts to overfit
            return train_losses, valid_losses
        else:
           last_valid_loss = valid_loss
        # if use_scheduler:
        #     scheduler.step(valid_loss) # for ReduceLROnPlateau
    if use_scheduler:
        scheduler.step()  

    return train_losses, valid_losses

def train(model, loader, loss_fn, optimizer):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(loader, 1):
        # print(f"X: {X.shape}")
        # print(f"y: {y.shape}")
        optimizer.zero_grad()
        pred = model(X)
        # print(f"pred: {pred.shape}")
        # print(f"y: {y.shape}")
        # pred = torch.squeeze(pred, dim=1)
        # print(f"pred1: {pred.shape}")
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    train_loss /= num_batches
    print(f"Train loss: {train_loss:>7f}")
    return train_loss

def validate(model, loader, loss_fn):
    num_batches = len(loader)
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            pred = torch.squeeze(pred, dim=1)
            loss = loss_fn(pred, y)
            valid_loss += loss.item()
    valid_loss /= num_batches
    print(f"Test loss: {valid_loss:>8f}")
    return valid_loss

def predict(model, loaders):
    model.eval()
    predictions = np.array([])
    original = np.array([])
    with torch.no_grad():
        for loader in loaders:
            for X, y in loader:
                pred = model(X)
                # pred = torch.squeeze(pred, dim=1)
                predictions = np.append(predictions, pred.numpy())
                original = np.append(original, y.numpy())
    return original, predictions