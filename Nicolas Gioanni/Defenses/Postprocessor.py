# Implementation for Postprocesser Defense: Random Noise (Chandrasekaranet al., 2018)

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from art.estimators.classification import KerasClassifier
from art.defences.postprocessor import GaussianNoise

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a simple Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=3, 
          validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy before defense: {test_acc * 100:.2f}%')

# Create an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

# Initialize the Gaussian Noise postprocessor
gaussian_noise = GaussianNoise(scale=0.1)

# Apply the postprocessor to the model's predictions
predictions = classifier.predict(test_images)
predictions_noisy = gaussian_noise(predictions)

# Evaluate the model's accuracy after applying the postprocessor
accuracy = np.mean(np.argmax(predictions_noisy, axis=1) == test_labels.flatten())
print(f'Accuracy after applying Gaussian Noise postprocessor: {accuracy * 100:.2f}%')
