#!/usr/bin/env python3

# ======================
# Import dependencies
# ======================
import tensorflow as tf
import tensorflow_datasets as tfds  # tensorflow datasets
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

# Initial setups
tfds.disable_progress_bar()  # to disable the progress bar
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# ======================
# Functions
# ======================
def normalize(images, labels):
    # This function will normalize the pixel values from 0-255 to 0-1
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def plot_image(i, predictions_array, true_labels, images, class_names):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid = False
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
    plt.grid = False
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    # Import the Fashion MNIST dataset
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    # Store class names
    class_names = metadata.features['label'].names
    print("Class names: {}".format(class_names))

    # Explore the data
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print('Number of train examples = ' + str(num_train_examples))
    print('Number of test examples = ' + str(num_test_examples))

    # Process the data
    # The map function applies the normalize function to each element in the train and test datasets
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    # Caching will keep the datasets in memory, making training faster
    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()

    # Explore the processed data
    # Take a single image, and remove the color dimension by reshaping
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28, 28))

    # Plot the image
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid = False
    plt.show()

    # Display the first 25 images from training
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_dataset.take(25)):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid = False
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()

    # ======================
    # Build the model
    # ======================
    # Setup the layers
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
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

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

    # Evaluate accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
    print('Accuracy on test dataset:', test_accuracy)

    # Making predictions and explore
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    predictions.shape
    predictions[0]
    np.argmax(predictions[0])
    test_labels[0]

    # We can graph this to look at the full set of 10 class predictions
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
    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = np.array([img])
    print(img.shape)

    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    np.argmax(predictions_single[0])


if __name__ == '__main__':
    main()




