import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Funktion zur Datenvorbereitung
def prepare_data(X_train, y_train, X_test, y_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return X_train, y_train, X_test, y_test

# Funktion zum Hinzufügen von Schichten zum MLP
def build_model(input_shape, layers):
    model = Sequential([InputLayer(input_shape=input_shape), Flatten()])
    for i, nodes in enumerate(layers):
        activation = 'relu' if i < len(layers) - 1 else 'softmax'
        model.add(Dense(nodes, activation=activation, name=f'fc{i + 1}'))
    return model

# Funktion zum Plotten der Trainingshistorie
def plot_history(history, title_text=''):
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    for key in ['loss', 'val_loss']:
        plt.plot(history.history[key], label=key)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    for key in ['accuracy', 'val_accuracy']:
        plt.plot(history.history[key], label=key)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(title_text)
    plt.show()

def main():
    # MNIST-Daten laden
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Daten vorbereiten
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # Modell erstellen
    input_shape = (28, 28)
    layers = (50,50,10)
    model = build_model(input_shape, layers)

    # Modell kompilieren
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Modell trainieren
    BATCH_SIZE = 128
    N_EPOCHS = 10
    train_history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test))

    # Trainingshistorie plotten
    plot_history(train_history, title_text='MNIST MLP [50 50 10]')

if __name__ == '__main__':
    main()
