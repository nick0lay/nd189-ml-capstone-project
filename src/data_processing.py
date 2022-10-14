import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def create_loaders(df, target_column, scalers, valid_days, test_days, train_len=7, target_len=1):
    print(f"Create loaders from {df.shape} with {target_column}")
    print(df.head())
    print(f"Scalers: {scalers}")
    print(f"Valid days: {valid_days}, test days: {test_days}, train lenght: {train_len}, target lenght: {target_len}")

    # split data
    valid_start_date = df.index[-valid_days-test_days]
    test_start_date = df.index[-valid_days]
    train, rest = split_data(df, valid_start_date)
    valid, test = split_data(rest, test_start_date)
    print(f"Train data shape: {train.shape}")
    print(f"Valid data shape: {valid.shape}")
    print(f"Test data shape: {test.shape}")

    # normalize data
    train_norm = normalize_data(train, scalers, True)
    valid_norm = normalize_data(valid, scalers)
    test_norm = normalize_data(test, scalers)
    print(f"Train norm data shape: {train_norm.shape}")
    print(f"Valid norm data shape: {valid_norm.shape}")
    print(f"Test norm data shape: {test_norm.shape}")

    # split data to sequences
    print(f"Start sequence split")
    X_train, y_train = split_to_sequences(train_norm, train_len, target_len, target_column)
    X_valid, y_valid = split_to_sequences(valid_norm, train_len, target_len, target_column)
    X_test, y_test = split_to_sequences(test_norm, train_len, target_len, target_column)

    # create loaders
    print(f"Create loaders")
    loader_train = create_loader(X_train, y_train)
    loader_valid = create_loader(X_valid, y_valid)
    loader_test = create_loader(X_test, y_test)
    return loader_train, loader_valid, loader_test


def create_loader(X, y, batch_size = 32, shuffle=False):
    x_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_to_sequences(df, train_len, target_len, target_column):
    X, y = [], []
    seq_len = train_len + target_len
    for i in range(seq_len, len(df) + 1):
        x_seq = df[i-seq_len:i-target_len]
        X.append(x_seq)

        y_seq = df[target_column][i-target_len:i]
        y.append(y_seq)
    return np.array(X), np.array(y)


def split_data(df, index):
    left_df = df[(df.index < index)]
    right_df = df[(df.index >= index)]
    return left_df, right_df


def normalize_data(df, column_normalizers, fit_normalizers=False):
    """
    Normalize feature columns for provided DataFrame.

    Return - DataFrame with normalized features.
    """
    features = pd.DataFrame()
    for columns, normalizer in column_normalizers:
        if normalizer is not None:
            print(f"Transform columns {columns}")
            if fit_normalizers:
                values = df[columns].to_numpy().flatten()
                normalizer.fit(values.reshape(-1, 1))
            for column in columns:
                data = df[column].values.reshape(-1, 1)
                normalized_data = normalizer.transform(data)
                features[column] = normalized_data.reshape(1, -1)[0]
        else:
            for column in columns:
                print(f"Copy column {column}")
                features[column] = df[column].values
    return features
        
