import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



# Function to add convolutional blocks
def add_cbb(model, cbb, block_name):
    for i, filters in enumerate(cbb):
        for j in range(len(filters)):
            model.add(Conv2D(filters[j], kernel_size=(3, 3), strides=(1, 1),
                             padding='same', activation='relu',
                             name=f'{block_name}_conv{j + 1}'))
        model.add(MaxPooling2D(pool_size=(2, 2), name=f'{block_name}_pool'))
    return model

# Function to plot training and validation history
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

# Set parameters
BATCH_SIZE = 32
N_EPOCHS = 5
TARGET_SIZE = (224, 224, )  # Adjust based on ImageNette size
PATH = "/home/user/Desktop/imagenette2"  # Update with your actual path

# Image data generators
imagegen_train = ImageDataGenerator(rescale=1./255.)
imagegen_val = ImageDataGenerator(rescale=1./255.)

# Load data using flow_from_directory
train = imagegen_train.flow_from_directory(
    f"{PATH}/train/",
    class_mode="categorical",
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE
)

val = imagegen_val.flow_from_directory(
    f"{PATH}/val/",
    class_mode="categorical",
    shuffle=False,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE
)

# Define input shape and CBB
INPUT = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
CBB = ((32,),(64,),(128,))  # Example CBB configuration

# Create an empty sequential network
model = Sequential()
model.add(InputLayer(input_shape=INPUT))

# Create convolutional blocks
block_count = 4
for filters in CBB:
    model = add_cbb(model, (filters,), f'block{block_count}')
    block_count += 1

# Add flattening and dense layers
model.add(Flatten())
model.add(Dense(32, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))

# Print model topology
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

# Train the model
train_history = model.fit(train, epochs=N_EPOCHS, validation_data=val)

# Plot training history
plot_history(train_history, 'ImageNette CNN')
