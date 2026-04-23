from tensorflow.keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

import os
os.environ['KERAS_HOME'] = './my_data'

from tensorflow.keras.datasets import mnist
mnist.load_data()