import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical


def prepare_data(X_train, y_train, X_test, y_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return X_train, y_train, X_test, y_test

 
def add_cbb(model, cbb, block_name):
    for i, filters in enumerate(cbb):
        for j in range(len(filters)):
            model.add(Conv2D(filters[j], kernel_size=(3, 3), strides=(1, 1),
                             padding='same', activation='relu',
                             name=f'{block_name}_conv{j + 1}'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=f'{block_name}_pool'))
    return model


# Plots training and validation history of network training
def plot_history(train_history, title_text=''):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.title(title_text + ' - Loss')
    
    plt.subplot(1, 2, 2)
    accuracy = train_history.history['accuracy']
    val_accuracy = train_history.history['val_accuracy']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title(title_text + ' - Accuracy')
    
    plt.show()


# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Prepare data for processing through the network
X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

# Define the input shape and CBB
INPUT = (32, 32, 3)  # CIFAR-10 images
CBB = ((30, 30), (30, 60, 90), (50,50,50))  # Example CBB configuration

# Create an empty sequential network
model = Sequential()
model.add(InputLayer(input_shape=INPUT))

# Create convolutional blocks
block_count = 1
for filters in CBB:
    model = add_cbb(model, (filters,), f'block{block_count}')
    block_count += 1

# Add flattening and dense layers
model.add(Flatten())
model.add(Dense(64, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))

# Print model topology
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

BATCH_SIZE = 32
N_EPOCH = 10
train_history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH,
                          validation_data=(X_test, y_test))

# Plot training history
plot_history(train_history, 'CIFAR-10 CNN')
