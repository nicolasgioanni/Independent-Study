# Implementation for Evasion Attack: DeepFool (Moosavi-Dezfooli et al., 2015)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from art.attacks.evasion import DeepFool
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_dataset

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('fashion_mnist')
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32")
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32")

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, num_classes=10).reshape(-1, 10)
y_test = to_categorical(y_test, num_classes=10).reshape(-1, 10)

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

# Wrap the model with ART classifier
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    input_shape=input_shape,
    nb_classes=10,
    clip_values=(0, 1)
)

# DeepFool Attack
deepfool = DeepFool(classifier=classifier)
#x_test_adv_deepfool = deepfool.generate(x=x_test)

# 10 samples for quicker testing
x_test_sample = x_test[:10]  
x_test_adv_deepfool = deepfool.generate(x=x_test_sample)

# Calculate accuracy on adversarial examples
# accuracy_deepfool = np.sum(np.argmax(classifier.predict(x_test_adv_deepfool), axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
# print(f"DeepFool Attack Accuracy: {accuracy_deepfool * 100:.2f}%")

# Limit y_test to the first 10 labels to match the number of adversarial examples for quicker testing
y_test_sample = y_test[:10]
# Calculate accuracy by comparing the predictions of adversarial examples to the first 10 labels in y_test
accuracy_deepfool = np.sum(np.argmax(classifier.predict(x_test_adv_deepfool), axis=1) == np.argmax(y_test_sample, axis=1)) / 10
print(f"DeepFool Attack Accuracy on Sample: {accuracy_deepfool * 100:.2f}%")

# Save DeepFool adversarial examples
np.save("x_test_adv_deepfool.npy", x_test_adv_deepfool)