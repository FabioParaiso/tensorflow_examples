import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
print(time.time())

NAME = f"Mnist-cnn-32x2-{int(time.time())}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

"Loads the mnist database"
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"Normalizes the inputs"
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"Reshapes the inputs"
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test = np.array(x_test).reshape(-1, 28, 28, 1)

"Creates the model"
model = Sequential()

"Creates the first layer"
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

"Creates the second layer"
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

"Creates a flatten layer"
model.add(Flatten())

"Creates a Dense layer"
model.add(Dense(32))
model.add(Activation('relu'))

"Creates the prediction layer"
model.add(Dense(10))
model.add(Activation('softmax'))

"Compiles the model"
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"Fits the model"
model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[tensorboard])

"Evaluates the model using the test data"
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
