import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


def train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=120,
                model_save_path='best_model.keras'):
    """Trains the model and saves the best model."""
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', mode='max', save_best_only=True,
                                       verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
                        verbose=1, callbacks=[early_stopping, model_checkpoint])
    return history


def plot_history(history):
    """Plots training history."""
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
