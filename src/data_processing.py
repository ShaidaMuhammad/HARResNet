import numpy as np
import pandas as pd
from scipy import stats


def read_data(file_path):
    """Reads data from a file and returns a cleaned DataFrame."""
    column_names = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis', 'a1', 'a2','a3', 'a4', 'a5', 'a6']
    df = pd.read_csv(file_path, header=None, names=column_names)
    df.drop(['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], axis=1, inplace=True)
    df['z-axis'] = df['z-axis'].str.replace(';', '').astype(float)
    df.dropna(axis=0, how='any', inplace=True)
    df["z-axis"] = df['z-axis'].astype(float)
    df.dropna(axis=0, how='any', inplace=True)

    return df


def show_basic_dataframe_info(dataframe):
    """Prints basic information about the DataFrame."""
    print(f'Number of columns in the dataframe: {dataframe.shape[1]}')
    print(f'Number of rows in the dataframe: {dataframe.shape[0]}\n')


def create_segments_and_labels(df, time_steps, step, label_name):
    """Creates segments and labels from the DataFrame."""
    num_features = 3
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xs, ys, zs])
        labels.append(label)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, num_features)
    labels = np.asarray(labels)
    return reshaped_segments, labels
