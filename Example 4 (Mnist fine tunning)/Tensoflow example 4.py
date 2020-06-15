import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time

"Loads the mnist database"
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"Normalizes the inputs"
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"Reshapes the inputs"
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test = np.array(x_test).reshape(-1, 28, 28, 1)


dense_layers = [0, 1, 2]
layer_sizes = [16, 32, 64]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

            "Creates the model"
            model = Sequential()

            "Creates the first layer"
            model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            "Creates the second and posterior layers"
            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            "Creates a flatten layer"
            model.add(Flatten())

            "Creates a Dense layers"
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            "Creates the prediction layer"
            model.add(Dense(10))
            model.add(Activation('softmax'))

            "Compiles the model"
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            "Fits the model"
            model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.1,
                      callbacks=[tensorboard])

