#!/usr/bin/env python3

# This exercise is very similar to the last one, the only difference being in the definition of the neural network
# layers, here we will use a convolutional neural network, but the principle is the same

# ===============================
# IMPORT LIBRARIES
# ===============================
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from colorama import Fore

# ===============================
# INITIAL SETUP
# ===============================
tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# ===============================
# FUNCTIONS
# ===============================
def normalize(images, labels):
    # Function to normalize pix values between 0 and 1
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# Functions for visualization
def plot_image(i, predictions_array, true_labels, images, class_names):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Import the Fashion MNIST dataset
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    print(Fore.BLUE + 'Dataset loaded' + Fore.RESET)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Explore the data
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print(Fore.BLUE + "Number of training examples: {}".format(num_train_examples) + Fore.RESET)
    print(Fore.BLUE + "Number of test examples:     {}".format(num_test_examples) + Fore.RESET)

    # Process the data
    # The map function applies the normalize function to each element
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    # Keep images on disk
    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()

    # Export the processed data
    # Take a single image, and remove the color dimension by reshaping
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28, 28))

    # Plot the image
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Display the first 25 images
    plt.figure(figsize=(10, 10))
    i = 0
    for (image, label) in test_dataset.take(25):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
        i += 1
    plt.show()

    # ========================
    # Build the model
    # ========================

    # Setup the layers
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    BATCH_SIZE = 32
    train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)
    model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

    # Evaluate accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
    print(Fore.BLUE + 'Accuracy on test dataset:' + str(test_accuracy) + Fore.RESET)

    # Make predictions and explore
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    print(Fore.BLUE + str(predictions.shape) + Fore.RESET)
    print(Fore.BLUE + str(predictions[0]) + Fore.RESET)
    print(Fore.BLUE + str(np.argmax(predictions[0])) + Fore.RESET)
    print(Fore.BLUE + str(test_labels[0]) + Fore.RESET)

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)

    # Grab an image from the test dataset
    img = test_images[0]
    print(Fore.BLUE + str(img.shape) + Fore.RESET)

    # Add the image to a batch where it's the only member.
    img = np.array([img])
    print(Fore.BLUE + str(img.shape) + Fore.RESET)

    predictions_single = model.predict(img)
    print(Fore.BLUE + str(predictions_single) + Fore.RESET)

    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)

    np.argmax(predictions_single[0])


if __name__ == '__main__':
    main()
