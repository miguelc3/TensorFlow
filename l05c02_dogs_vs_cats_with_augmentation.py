#!/usr/bin/env python3

# Font: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c02_dogs_vs_cats_with_augmentation.ipynb

# ==============================
# IMPORT LIBRARIES
# ==============================
import tensorflow as tf
from colorama import Fore
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# FUNCTIONS
# ==============================
def plotImages(images_arr):
    # This function will plot images in the form of a grid with 1 row and 5 columns
    # where images are placed in each column.
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# Main function
def main():
    # Basic setups
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # Data Loading
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

    # assign variables with the proper file path for the training and validation sets
    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    # Understanding the data
    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))
    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))
    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    # Print number of data
    print(Fore.BLUE + 'total training cat images:' + str(num_cats_tr))
    print('total training dog images:' + str(num_dogs_tr))
    print('total validation cat images:' + str(num_cats_val))
    print('total validation dog images:' + str(num_dogs_val))
    print("--")
    print("Total training images:" + str(total_train))
    print("Total validation images:" + str(total_val) + Fore.RESET)

    # Setting model parameters
    BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
    IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

    # Data Augmentation - Flipping the image horizontally
    image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))

    # See transformation in action
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)

    # Rotating the image
    image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)

    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))

    # See transformation in action
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)

    # Applying zoom
    image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)

    train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE))

    # See transformation in action
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)

    # Putting it all together
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                         class_mode='binary')

    #  visualize how a single image would look like five different times
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)

    # Creating Validation Data generator
    image_gen_val = ImageDataGenerator(rescale=1. / 255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=validation_dir,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

    # ==============================
    # MODEL CREATION
    # ==============================
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train model
    # Since the batches are coming from a generator(ImageDataGenerator) we will use fit_generator instead od fit
    EPOCHS = 60
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )

    # Visualizing results of the training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()


if __name__ == '__main__':
    main()
