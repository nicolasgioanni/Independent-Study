# Implementation for Detector Defense: 5.1 Evasion, Basic detector based on inputs

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from art.defences.detector.evasion import BinaryInputDetector

# Step 1: Load and preprocess your dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Step 2: Train a simple classifier on the clean training data
classifier_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
classifier_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
classifier_model.fit(x_train, y_train, epochs=5, batch_size=64)

# Wrap the model with ART's TensorFlowV2Classifier
classifier = TensorFlowV2Classifier(
    model=classifier_model,
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=classifier_model.optimizer,
    input_shape=(28, 28, 1),
    nb_classes=10,
    clip_values=(0, 1)
)

# Step 3: Generate adversarial samples using an attack method
attack = FastGradientMethod(estimator=classifier, eps=0.3)
adv_samples = attack.generate(x=x_train)

# Step 4: Combine clean and adversarial samples to create the detector's training set
clean_samples = x_train
x_detector_train = np.concatenate((clean_samples, adv_samples), axis=0)
y_detector_train = np.concatenate((np.zeros(len(clean_samples)), np.ones(len(adv_samples))), axis=0)

# Step 5: Define the detector model
detector_model = models.Sequential([
    layers.Flatten(input_shape=x_detector_train.shape[1:]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
detector_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Step 6: Train the detector model
detector_model.fit(x_detector_train, y_detector_train, epochs=10, batch_size=64)

# Step 7: Initialize the BinaryInputDetector with the trained detector model
detector = BinaryInputDetector(detector_model)

# Step 8: Detect adversarial samples in the test set
_, is_adversarial = detector.detect(x_test)
print(is_adversarial)  # Boolean array indicating which samples are detected as adversarial
