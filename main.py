import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from src.data_processing import read_data, show_basic_dataframe_info, create_segments_and_labels
from src.models import build_inception_model
from src.train import train_model, plot_history
from src.utils import plot_confusion_matrix, plot_classification_report

# Constants
TIME_PERIODS = 80
STEP_DISTANCE = 40
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']


def main():
    # Load and preprocess data
    df = read_data('data/WISDM_ar_v1.1_raw.txt')
    show_basic_dataframe_info(df)
    df['activity'] = df['activity'].str.strip()
    le = LabelEncoder()
    df['ActivityEncoded'] = le.fit_transform(df['activity'].values.ravel())
    df['x-axis'] /= df['x-axis'].max()
    df['y-axis'] /= df['y-axis'].max()
    df['z-axis'] /= df['z-axis'].max()
    df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

    # Create segments and labels
    x_data, y_data = create_segments_and_labels(df, TIME_PERIODS, STEP_DISTANCE, 'ActivityEncoded')

    # Split data into training, testing, and cross-validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    X_cross, X_test, y_cross, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # One-hot encode labels
    num_classes = le.classes_.size
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_cross = to_categorical(y_cross, num_classes)

    # Build and train model
    model = build_inception_model(X_train.shape[1:], num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = train_model(model, X_train, y_train, X_cross, y_cross)

    # Plot training history
    plot_history(history)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix and classification report
    y_pred = model.predict(X_test).argmax(axis=1)
    plot_confusion_matrix(y_test.argmax(axis=1), y_pred, LABELS)
    plot_classification_report(y_test.argmax(axis=1), y_pred, LABELS)


if __name__ == '__main__':
    main()
