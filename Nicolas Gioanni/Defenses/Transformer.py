# Implementation for Transformer Defense: 4.1 Evasion, Defensive Distillation (Papernot et al., 2015)​

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model

# Set the temperature
temperature = 100.0

# Custom softmax function with temperature
def softmax_with_temperature(logits, temperature):
    return tf.nn.softmax(logits / temperature)

# Train the initial model with elevated temperature
initial_model = create_model()
initial_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
initial_model.fit(x_train, y_train, epochs=5, batch_size=64)

# Generate soft labels
logits = initial_model.predict(x_train)
soft_labels = softmax_with_temperature(logits, temperature)

# Train the distilled model using soft labels
distilled_model = create_model()
distilled_model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
distilled_model.fit(x_train, soft_labels, epochs=5, batch_size=64)

# Evaluate the distilled model
distilled_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
test_loss, test_acc = distilled_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
