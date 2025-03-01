# Implementation for Preprocesser Defense: JPEG Compression (Dziugaite et al., 2016)

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from art.defences.preprocessor import JpegCompression
from art.estimators.classification import KerasClassifier

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model for 3 epochs
model.fit(train_images, train_labels, epochs=3, 
          validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy before defense: {test_acc * 100:.2f}%')

# Create an ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Initialize JPEG Compression defense
jpeg_compression = JpegCompression(clip_values=(0, 1), quality=50)

# Apply the defense to the test set
test_images_defended, _ = jpeg_compression(test_images)

# Evaluate the classifier on the defended test set
predictions = classifier.predict(test_images_defended)
accuracy = np.mean(np.argmax(predictions, axis=1) == test_labels.flatten())
print(f'Accuracy on defended test set: {accuracy * 100:.2f}%')
