import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    # Initialize lists for storing image data and labels
    images = []
    labels = []

    # Iterate over the subdirectories in `data_dir`
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Check if the current folder is a directory
        if os.path.isdir(folder_path):

            # Iterate over the images in the subdirectory
            for image_file in os.listdir(folder_path):

                # Read the image file and resize it to IMG_WIDTH x IMG_HEIGHT
                image = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

                # Append the image data and its label (which is the folder name)
                images.append(image)
                labels.append(int(folder))

    # Return the image data and labels
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    # Create a sequential model
    model = tf.keras.models.Sequential([
        # Add a 2D convolution layer with 32 filters of size (3,3), relu activation function 
        # and appropriate input shape
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Add a MaxPooling layer with pool size (2,2) 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Add another 2D convolution layer with 64 filters of size (3,3) and relu activation function
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),

        # Add another MaxPooling layer with pool size (2,2) 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the tensor output from the previous layer
        tf.keras.layers.Flatten(),

        # Add a Dense hidden layer with 128 neurons and relu activation function
        tf.keras.layers.Dense(128, activation="relu"),

        # Add another Dense hidden layer with 64 neurons and relu activation function
        tf.keras.layers.Dense(64, activation="relu"),

        # Apply dropout to prevent overfitting, here approx 33% of the neurons will be turned off during training 
        tf.keras.layers.Dropout(0.33),

        # Add output Dense layer with NUM_CATEGORIES neurons (each for one category) 
        # with softmax activation function for multiclass classification
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model with Adam optimizer and categorical cross entropy as loss function 
    # and accuracy as metrics to observe
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Return the compiled model
    return model

if __name__ == "__main__":
    main()