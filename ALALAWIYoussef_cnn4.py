#%% Import necessary libraries
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Define the folder containing the images
folder = "ds_final_assignment"

# Create a list to store the images and labels
images = []
labels = []

# Iterate through the images in the folder
for filename in os.listdir(folder):
    # Extract the label from the filename
    label = filename.split(".")[0]

    # Open the image, remove the alpha channel and resize it to 50x50 pixels
    img = Image.open(os.path.join(folder, filename))
    img = img.convert('RGB')
    img = img.resize((50, 50))

    # Add the image and label to the list
    images.append(np.array(img))
    labels.append(label)

#%% Convert the images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

#%% Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

#%% Convert the labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

#%% Define the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4, activation='softmax'))

#%% Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% Fit the model
model.fit(X_train, y_train, epochs=100)

#%% Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

#%% Make predictions on the test set
y_pred = model.predict(X_test)

#%% Generate a classification report
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred.argmax(axis=1))
print(classification_report(y_test_original, y_pred_original))


#%% Generate a heatmap of the confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred.argmax(axis=1)).numpy()
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
