from tensorflow.keras.datasets import mnist
import os
os.environ['KERAS_HOME'] = '/Users/nguyenvokhang/Downloads/MNIST'
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
