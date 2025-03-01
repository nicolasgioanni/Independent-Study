# Implementation for Poisoning Attack: Backdoor Attack (Gu et. al., 2017)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_dataset

# Load the MNIST dataset (for simplicity, as ImageNet is computationally intensive)
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32")
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32")

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = np.argmax(to_categorical(y_train, 10), axis=1)
y_test = np.argmax(to_categorical(y_test, 10), axis=1)

# Define the model architecture
def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model
input_shape = (28, 28, 1)
model = create_model(input_shape)
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Wrap the model with ART's classifier
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    input_shape=input_shape,
    nb_classes=10,
    clip_values=(0, 1)
)

# Define a simple trigger for the backdoor attack
#trigger = np.zeros((28, 28, 1), dtype=np.float32)
#trigger[25:28, 25:28, :] = 1.0  # A 3x3 white square in the corner as the trigger

# Define a function that applies the trigger to an input image
def trigger_function(x):
    x[25:28, 25:28, :] = 1.0  # A 3x3 white square in the corner as the trigger
    return x

# Define the backdoor attack with the trigger function
backdoor_attack = PoisoningAttackBackdoor(perturbation=trigger_function)


# Poison part of the training data
#poisoned_data, poisoned_labels = backdoor_attack.poison(x_train[:5000], y_train[:5000])
poisoned_data, poisoned_labels = backdoor_attack.poison(x_train[:10], y_train[:10])

# Retrain the model with poisoned data
#x_train_combined = np.concatenate([x_train, poisoned_data])
#y_train_combined = np.concatenate([y_train, poisoned_labels])
x_train_combined = np.concatenate([x_train[:10], poisoned_data])
y_train_combined = np.concatenate([y_train[:10], poisoned_labels])

# Train with the combined dataset
#model.fit(x_train_combined, y_train_combined, epochs=5, batch_size=128, validation_split=0.1)
model.fit(x_train_combined, y_train_combined, epochs=5, batch_size=10, validation_split=0.1)

# Evaluate the model on clean and poisoned test data
clean_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"Clean Test Accuracy: {clean_accuracy * 100:.2f}%")

# Add trigger to a portion of the test set to simulate backdoor activation
#x_test_backdoor = x_test.copy()
#x_test_backdoor[:, 25:28, 25:28, :] = 1.0  # Apply the trigger to test samples
x_test_backdoor = x_test[:10].copy()
x_test_backdoor[:, 25:28, 25:28, :] = 1.0  # Apply the trigger to the first 10 test samples


# Evaluate model performance on backdoor examples
#backdoor_accuracy = model.evaluate(x_test_backdoor, y_test, verbose=0)[1]
backdoor_accuracy = model.evaluate(x_test_backdoor, y_test[:10], verbose=0)[1]

print(f"Backdoor Attack Test Accuracy: {backdoor_accuracy * 100:.2f}%")