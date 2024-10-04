import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import convert_to_tensor
from skimage.transform import resize
from tensorflow.keras.applications import ResNet50

MODEL_FILE = '/home/user/Desktop/Tf/U6/model/'  # Adjust the path as needed
IMAGE_PATH_AND_FN = '/home/user/Desktop/imagenette2/val/n03394916/ILSVRC2012_val_00033682.JPEG'  # Adjust to your image path
PATCH_SIZE = 32

model = load_model(MODEL_FILE)

model = ResNet50(weights='imagenet', include_top=True)

img = load_img(IMAGE_PATH_AND_FN, target_size=(224, 224))  # Use the target size from training
img = img_to_array(img) / 255.0  # Normalize the image
tensor_img = convert_to_tensor(np.array([img]), dtype='float32')

# Get the original class confidence
pred_classes = model.predict(tensor_img)
pred_class = np.argmax(pred_classes[0])
pred_conf = np.max(pred_classes[0])

# Prepare patched images
SX, SY = img.shape[0], img.shape[1]
nmb_patches = (SX // PATCH_SIZE) * (SY // PATCH_SIZE)
patched_images = np.zeros((nmb_patches, SX, SY, 3))

# Loop to create patched images
patch_idx = 0
for i in range(0, SX, PATCH_SIZE):
    for j in range(0, SY, PATCH_SIZE):
        patched_img = np.copy(img)
        patched_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] = 128 / 255.0  # Gray value
        patched_images[patch_idx] = patched_img
        patch_idx += 1

# Convert patched images to tensor
patched_images_tensor = convert_to_tensor(patched_images, dtype='float32')

# Predict classes for patched images
pred_classes_patched = model.predict(patched_images_tensor)

# Create the sensitivity map
sensitivity_map = np.zeros((SX // PATCH_SIZE, SY // PATCH_SIZE))
for i in range(nmb_patches):
    sensitivity_map[i // (SY // PATCH_SIZE), i % (SY // PATCH_SIZE)] = pred_conf - pred_classes_patched[i, pred_class]

# Resize sensitivity map to original image size
sensitivity_map_resized = resize(sensitivity_map, (SX, SY), anti_aliasing=True)

#Visualization
plt.imshow(img)
plt.imshow(sensitivity_map_resized, cmap='jet', alpha=0.5)
plt.axis("off")
plt.colorbar()
plt.show()
