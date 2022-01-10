#!/usr/bin/env python3

# Font: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c01_tensorflow_hub_and_transfer_learning.ipynb#scrollTo=FVM2fKGEHIJN

# ===========================================
# IMPORTS
# ===========================================
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow.keras as layers
import logging
import numpy as np
import PIL.Image as Image
import cv2


# global variables
IMAGE_RES = 224

# ===========================================
# FUNCTIONS
# ===========================================
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label


# ===========================================
# MAIN FUNCTION
# ===========================================
def main():
    # Initial setups
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    (train_examples, validation_examples), info = tfds.load(
        'cats_vs_dogs',
        with_info=True,
        as_supervised=True,
        split=['train[:80%]', 'train[80%:]'],
    )

    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes

    BATCH_SIZE = 32

    train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

    image_batch, label_batch = next(iter(train_batches.take(1)))
    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes

    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    feature_extractor = hub.KerasLayer(URL,
                                       input_shape=(IMAGE_RES, IMAGE_RES, 3))

    feature_batch = feature_extractor(image_batch)
    print(feature_batch.shape)

    feature_extractor.trainable = False

    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(2)
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    EPOCHS = 6
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)


if __name__ == '__main__':
    main()







