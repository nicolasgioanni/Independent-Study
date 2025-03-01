# Implementation for Inference Attack: Model Inversion MIFace (Fredrikson et al., 2015)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from art.attacks.inference.model_inversion import MIFace
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_dataset
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32")
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32")
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ensure labels are one-hot encoded with correct shape (batch_size, 10)
if y_train.ndim == 1 or y_train.shape[-1] != 10:
    y_train = to_categorical(y_train, 10)
if y_test.ndim == 1 or y_test.shape[-1] != 10:
    y_test = to_categorical(y_test, 10)

# After to_categorical, print shapes
print("Shape of y_train after encoding:", y_train.shape)
print("Shape of y_test after encoding:", y_test.shape)

# Reshape if there's an extra dimension
if y_train.shape == (60000, 10, 10):
    y_train = y_train.reshape((60000, 10))
if y_test.shape == (10000, 10, 10):
    y_test = y_test.reshape((10000, 10))

print("Final shape of x_train:", x_train.shape)
print("Final shape of y_train:", y_train.shape)
print("Final shape of x_test:", x_test.shape)
print("Final shape of y_test:", y_test.shape)

# Define a simple model architecture
def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (28, 28, 1)
model = create_model(input_shape)
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# Wrap the model with ART’s TensorFlowV2Classifier
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    input_shape=input_shape,
    nb_classes=10,
    clip_values=(0, 1)
)

# Initialize the Model Inversion attack (MIFace)
mi_attack = MIFace(classifier, max_iter=100, learning_rate=1.0)

# Run the attack for a specific target class
target_class = 3  # Specify the class you want to invert
reconstructed_image = mi_attack.infer(x=None, y=np.array([target_class]))

# Display the reconstructed image (requires matplotlib)
plt.imshow(reconstructed_image[0].reshape(28, 28), cmap='gray')
plt.title(f"Reconstructed Image for Class {target_class}")
plt.show()
