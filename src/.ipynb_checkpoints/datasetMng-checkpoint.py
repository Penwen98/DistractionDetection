import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical



# Set the path to the dataset
data_path = sys.path[0] + "/Distracted Detection Dataset"

# Get the list of animal categories
categories = os.listdir(data_path)

# Create a list to store the image filenames and labels
data = []
labels = []

# Load the images and labels
for category in categories:
    folder_path = os.path.join(data_path, category)
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        data.append(image_path)
        labels.append(category)

# Split the data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)


# Define a function to preprocess the images
def preprocess_image(image_path):
    # Load the image and resize it to (150, 150)
    img = load_img(image_path, target_size=(48, 48))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Scale the pixel values to the range of [0, 1]
    img_array /= 255.0
    img_array = img_array.reshape(128, 54, 1)
    # Return the preprocessed image
    return img_array


# Define a dictionary that maps category strings to integer labels
label_map = {category: i for i, category in enumerate(categories)}

# Load the images and labels
for category in categories:
    folder_path = os.path.join(data_path, category)
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        data.append(image_path)
        labels.append(label_map[category])

global y_train, y_test, X_train, X_test

# Convert the integer labels to one-hot encoded vectors
y_train = to_categorical(np.array([label_map[label] for label in train_labels]))
y_test = to_categorical(np.array([label_map[label] for label in test_labels]))


# Preprocess the images in the train set
X_train = np.array([preprocess_image(image_path) for image_path in train_data])
#Preprocess the images in the test set
X_test = np.array([preprocess_image(image_path) for image_path in test_data])


