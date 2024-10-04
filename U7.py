import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dense, Add,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from PIL import ImageFile
from tensorflow.keras.mixed_precision import experimental as mixed_precision


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# Function to create a residual block
def residual_block(inputs, filters, reduce_resolution=False):
    if reduce_resolution:
        x1 = Conv2D(filters, kernel_size=(3, 3), strides=(2, 2),
                     padding='same', activation='relu')(inputs)
        x2 = Conv2D(filters, kernel_size=(3, 3), padding='same',
                     activation='relu')(x1)
        bypass = Conv2D(filters, kernel_size=(1, 1), strides=(2, 2),
                        padding='same')(inputs)
    else:
        x1 = Conv2D(filters, kernel_size=(3, 3), padding='same',
                     activation='relu')(inputs)
        x2 = Conv2D(filters, kernel_size=(3, 3), padding='same',
                     activation='relu')(x1)
        bypass = inputs
    
    outputs = Add()([x2, bypass])
    return outputs

# Function to create the ResNet architecture
def make_resnet(input_shape, rbb_blocks):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    
    for no_rbs, filters in rbb_blocks:
        for i in range(no_rbs):
            reduce_resolution = (i == 0)
            x = residual_block(x, filters, reduce_resolution)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Set parameters
BATCH_SIZE = 64
N_EPOCHS = 200
LEARNING_RATE = 0.0005
PATIENCE = 20
TARGET_SIZE = (128, 128)
PATH = "/home/user/Desktop/imagenette2"

# Image data generators with data augmentation
imagegen_train = ImageDataGenerator(
    rescale=1./255.,
    brightness_range=[0.9, 1.1],
    zoom_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    fill_mode='reflect'
)
imagegen_val = ImageDataGenerator(rescale=1./255.)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load data
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

INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
RBB_BLOCKS = ((4, 128), (6, 256), (3, 512))

# Create the ResNet model
model = make_resnet(INPUT_SHAPE, RBB_BLOCKS)

# Print model summary
model.summary()

# Compile the model
loss_function = CategoricalCrossentropy()
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss=loss_function, metrics=['accuracy'], optimizer=optimizer)

# Early stopping callback
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

# Train the model
train_history = model.fit(train, epochs=N_EPOCHS, validation_data=val, callbacks=[es])

# Save the trained model
model.save('/home/user/Desktop/Tf/U7/model/')  # Update the path as needed

# Function to plot training history
def plot_history(train_history, title_text=''):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'], label='Loss')
    plt.plot(train_history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(title_text + ' - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_history.history['accuracy'], label='Accuracy')
    plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(title_text + ' - Accuracy')

    plt.show()

# Plot training history
plot_history(train_history, 'ImageNette ResNet')
