import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical

from src.utils import plot_confusion_matrix, plot_classification_report
from src.data_processing import create_segments_and_labels


def load_and_preprocess_data(file_path, time_steps, step, label_name):
    """Loads and preprocesses the data."""
    df = pd.read_csv(file_path)
    df['z-axis'] = df['z-axis'].str.replace(';', '').astype(float)
    df.dropna(axis=0, how='any', inplace=True)
    le = LabelEncoder()
    df['ActivityEncoded'] = le.fit_transform(df[label_name].values.ravel())
    df['x-axis'] /= df['x-axis'].max()
    df['y-axis'] /= df['y-axis'].max()
    df['z-axis'] /= df['z-axis'].max()
    df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    x_data, y_data = create_segments_and_labels(df, time_steps, step, 'ActivityEncoded')
    num_classes = le.classes_.size
    y_data = to_categorical(y_data, num_classes)
    return x_data, y_data, le.classes_


def evaluate_model(model_path, test_data_path, time_steps=80, step=40, label_name='activity'):
    """Evaluates the model."""
    model = load_model(model_path)
    X_test, y_test, classes = load_and_preprocess_data(test_data_path, time_steps, step, label_name)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Predictions and evaluation metrics
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Plot confusion matrix and classification report
    plot_confusion_matrix(y_true, y_pred, classes)
    plot_classification_report(y_true, y_pred, classes)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the trained model on test data.')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('test_data_path', type=str, help='Path to the test data in CSV format')
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data_path)
