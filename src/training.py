import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

use_scheduler = True

def train_model(model, loader_train, loader_valid, epochs=30, lr=0.0001, seq_len=7, use_mask=True, device=None):
    hook = get_hook(create_if_not_exists=True)
    loss_fn = nn.MSELoss(reduction='mean') # default 'mean'
    if hook:
        hook.register_loss(loss_fn)
    # loss_fn = nn.L1Loss();
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    if use_scheduler:
        scheduler = ExponentialLR(optimizer, gamma=0.9) # 0.995
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8)

    # generate mask
    mask = None
    if use_mask:
        mask = generate_square_subsequent_mask(seq_len).to(device)

    train_losses = []
    valid_losses = []
    last_valid_loss = None
    min_epochs = 1000 # use 1000 to disable early stopping
    train_loss_target = 0.00155
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, loader_train, loss_fn, optimizer, mask, device=device)
        valid_loss = validate(model, loader_valid, loss_fn, mask, device=device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # early training stopping
        if min_epochs <= epoch and last_valid_loss < valid_loss or train_loss < train_loss_target:
            # exit model starts to overfit
            return train_losses, valid_losses
        else:
           last_valid_loss = valid_loss
        # if use_scheduler:
        #     scheduler.step(valid_loss) # for ReduceLROnPlateau
    if use_scheduler:
        scheduler.step() # for ExponentialLR 

    return train_losses, valid_losses

def train(model, loader, loss_fn, optimizer, mask, device=None):
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    size = len(loader.dataset)
    num_batches = len(loader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(loader, 1):
        if device:
            X = X.to(device)
            y = y.to(device)
        optimizer.zero_grad()
        pred = model(X, mask)
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

def validate(model, loader, loss_fn, mask, device=None):
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(smd.modes.EVAL)
    num_batches = len(loader)
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for X, y in loader:
            if device:
                X = X.to(device)
                y = y.to(device)
            pred = model(X, mask)
            loss = loss_fn(pred, y)
            valid_loss += loss.item()
    valid_loss /= num_batches
    print(f"Test loss: {valid_loss:>8f}")
    return valid_loss

def predict(model, loader, mask, device=None):
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(smd.modes.PREDICT)
    model.eval()
    predictions = np.array([])
    with torch.no_grad():
        for X, y in loader:
            if device:
                X = X.to(device)
                y = y.to(device)
                pred = model(X, mask)
                predictions = np.append(predictions, pred.numpy())
    return predictions

# Predict data for multiple loaders
def predict_multiple(model, loaders, mask, device=None):
    model.eval()
    predictions = np.array([])
    original = np.array([])
    with torch.no_grad():
        for loader in loaders:
            for X, y in loader:
                if device:
                    X = X.to(device)
                    y = y.to(device)
                pred = model(X, mask)
                predictions = np.append(predictions, pred.numpy())
                original = np.append(original, y.numpy())
    return original, predictions

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask