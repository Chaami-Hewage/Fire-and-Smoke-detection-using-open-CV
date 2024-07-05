import os
import cv2
import numpy as np
import scikitlearn as sk

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

training_data = [r'C:\Users\Asus\Desktop\Fire-and-Smoke-detection-using-open-CV\firedataset\fire_images',
                 r'C:\Users\Asus\Desktop\Fire-and-Smoke-detection-using-open-CV\firedataset\non_fire_images']


def load_images(data_dir):
    images = []  # Initialize empty list to store the images
    labels = []  # Initialize empty list to store the labels

    for i in range(len(training_data)):
        folder = training_data[i]  # Get the folder of the images
        label = i  # Get the label of the images
        for filename in os.listdir(folder):
            try:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Read the image file
                img = cv2.resize(img, (48, 48))  # Resize the image
                images.append(img)  # Append the image
                labels.append(label)  # Append the label
            except Exception as e:
                print(f"Error loading image: {os.path.join(folder, filename)}: {str(e)}")

    return np.array(images), np.array(labels)  # Return the images and labels as numpy arrays


images, labels = load_images(training_data)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)  # Split the data into training and testing sets
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1).astype('float32') / 255  # Reshape the data and normalize
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1).astype('float32') / 255

y_train = to_categorical(y_train)  # Convert the labels to categorical
y_test = to_categorical(y_test)

model = Sequential()  # Initialize the model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))  # Add the first convolution layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  # Add the second convolution layer

model.add(MaxPooling2D(pool_size=(2, 2)))  # Add the first pooling layer

model.add(Dropout(0.25))  # Add the first dropout layer
model.add(Flatten())  # Add the flatten layer
model.add(Dense(128, activation='relu'))  # Add the first dense layer
model.add(Dropout(0.5))  # Add the second dropout layer
model.add(Dense(2, activation='softmax'))  # Add the output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))  # Fit the model
model.save('fire_detection_model.h5')  # Save the model
