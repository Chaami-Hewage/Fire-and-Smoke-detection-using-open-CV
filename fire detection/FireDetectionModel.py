import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ensure keras is correctly imported from tensorflow
keras = tf.keras
models = keras.models
layers = keras.layers
utils = keras.utils

# Import necessary components from tensorflow.keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# Updated paths using raw strings
training_data = [r"C:\Users\Asus\Desktop\Fire-and-Smoke-detection-using-open-CV\fire detection\firedataset\fire_images",
                  r"C:\Users\Asus\Desktop\Fire-and-Smoke-detection-using-open-CV\fire detection\firedataset\non_fire_images"]

def load_images(data_dir):
    images = []
    labels = []

    for i in range(len(training_data)):
        folder = training_data[i]
        label = i
        print(f"Processing folder: {folder}")  # Debugging print statement
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            continue
        for filename in os.listdir(folder):
            try:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image: {os.path.join(folder, filename)}: {str(e)}")

    return np.array(images), np.array(labels)

images, labels = load_images(training_data)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))
model.save('fire_detection_model.h5')
