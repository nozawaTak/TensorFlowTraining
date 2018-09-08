import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist

input_data = mnist.input_data.read_data_sets("MNIST_data", one_hot=true)

print(input_data)
