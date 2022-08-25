import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Source: https://github.com/CVxTz/time_series_forecasting/blob/main/scripts/generate_time_series.py

periods = [7, 14, 28, 30]

def get_init_df():
    date_range = pd.date_range(start="2018-01-01", end="2020-01-01", freq="D")
    dataframe = pd.DataFrame(date_range, columns=["timestamp"])
    dataframe["index"] = range(dataframe.shape[0])
    return dataframe

def generate_time_series(df):
    clip_val = random.uniform(0.3, 1)
    period = random.choice(periods)
    phase = random.randint(-1000, 1000)

    df["views"] = df.apply(
        lambda x: np.clip(
            np.cos(x["index"] * 2 * np.pi / period + phase), -clip_val, clip_val
        )
        * x["amplitude"]
        + x["offset"],
        axis=1,
    ) + np.random.normal(
        0, df["amplitude"].abs().max() / 10, size=(df.shape[0],)
    )

    return df

def set_amplitude(dataframe):

    max_step = random.randint(90, 365)
    max_amplitude = random.uniform(0.1, 1)
    offset = random.uniform(-1, 1)

    phase = random.randint(-1000, 1000)

    amplitude = (
        dataframe["index"]
        .apply(lambda x: max_amplitude * (x % max_step + phase) / max_step + offset)
        .values
    )

    if random.random() < 0.5:
        amplitude = amplitude[::-1]

    dataframe["amplitude"] = amplitude

    return dataframe

def set_offset(dataframe):

    max_step = random.randint(15, 45)
    max_offset = random.uniform(-1, 1)
    base_offset = random.uniform(-1, 1)

    phase = random.randint(-1000, 1000)

    offset = (
        dataframe["index"]
        .apply(
            lambda x: max_offset * np.cos(x * 2 * np.pi / max_step + phase)
            + base_offset
        )
        .values
    )

    if random.random() < 0.5:
        offset = offset[::-1]

    dataframe["offset"] = offset

    return dataframe

def generate_df():
    dataframe = get_init_df()
    dataframe = set_amplitude(dataframe)
    dataframe = set_offset(dataframe)
    dataframe = generate_time_series(dataframe)
    return dataframe