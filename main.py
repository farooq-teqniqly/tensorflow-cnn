from tensorflow.keras.datasets import cifar10 as cf10
from tensorflow.keras.utils import to_categorical

# Load dataset
(train_images, train_labels), (test_images, test_labels) = cf10.load_data()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Normalize pixel values
MAX_PIXEL_VALUE = 255

train_images = train_images / MAX_PIXEL_VALUE
test_images = test_images / MAX_PIXEL_VALUE

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
