import numpy as np
import random
import pickle
import sys


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return (1.0 - sigmoid(z)) * sigmoid(z)


def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))


def tanh_derivative(z):
    return 1 - np.power(np.tanh(z), 2)


class NeuralNetwork:
    def initialize_weights(self):
        for layer in range(len(self.layers) - 1):
            # Add 1 for the bias too
            w = np.random.normal(0, 0.1, (self.layers[layer] + 1, self.layers[layer + 1]))
            self.weights.append(w)

    def __init__(self, layers):
        self.layers = layers
        self.weights = []

        self.activation_function = tanh
        self.activation_function_derivative = tanh_derivative

        self.initialize_weights()

    def forward(self, x):
        all_layers = [x]
        for i in range(len(self.weights) - 1):
            v = np.dot(all_layers[i], self.weights[i])
            v = self.activation_function(v)

            # add the bias for the next layer
            v = np.concatenate((np.ones(1), np.array(v)))
            all_layers.append(v)

        # last layer
        v = np.dot(all_layers[-1], self.weights[-1])
        v = self.activation_function(v)
        all_layers.append(v)
        return all_layers

    def backpropagation(self, y, all_layers_of_x):

        # Calculate error and delta for the last layer
        last_layer = all_layers_of_x[-1]
        error = y - last_layer
        deltas = [error * self.activation_function_derivative(last_layer)]

        # Calculate errors and deltas for the other layers
        for i in range(len(self.layers) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i][1:].T)
            error = error * self.activation_function_derivative(all_layers_of_x[i][1:])
            deltas.append(error)

        # Last layer delta should be in the end
        deltas.reverse()
        return deltas

    def train(self, data, labels, learning_rate=0.1, epochs=100, runs_per_epoch=15000):

        # Add biases for each data
        ones = np.ones((1, data.shape[0]))
        biased_data = np.concatenate((ones.T, data), axis=1)

        for i in range(epochs * runs_per_epoch):
            print "iteration: ", i

            # Get random x
            random_index = np.random.randint(data.shape[0])
            x = biased_data[random_index]
            y = labels[random_index]

            # Forward Propagation
            all_layers_for_current_x = self.forward(x)

            delta_vec = self.backpropagation(y, all_layers_for_current_x)

            self.update_rule(all_layers_for_current_x, delta_vec, learning_rate)

    def update_rule(self, all_layers, delta_vec, learning_rate):
        for i in range(len(self.weights)):
            layer = all_layers[i].reshape(1, self.layers[i] + 1)
            delta = delta_vec[i].reshape(1, self.layers[i + 1])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    def get_x_prediction(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activation_function(np.dot(val, self.weights[i]))
            val = np.concatenate((np.ones(1).T, np.array(val)))
        return val[1]


def load_date_from_file(file_name):
    with open(file_name, mode='rb') as file:
        data = []
        for line in file:
            data.append(line.split())
    data = np.array(data)

    X = [item[0] for item in data]
    y = [int(item[1]) for item in data]

    pure_x = []
    for x in X:
        temp = []
        x = list(x)
        for char in x:
            temp.append(int(char))
        pure_x.append(temp)

    return pure_x, y


def shuffle_arrays(arr1, arr2):
    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    arr1, arr2 = zip(*combined)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    return arr1, arr2


def get_prediction_class(prediction):
    if prediction < 0.5:
        return 0
    return 1


def calculate_success(validation_x, validation_y):
    success = 0
    for x, y in zip(validation_x, validation_y):
        prediction = nn.get_x_prediction(x)
        prediction = get_prediction_class(prediction)
        if y == prediction:
            success += 1

    return float(success) / len(validation_y) * 100


train_filename = sys.argv[1]
validation_filename = sys.argv[2]

# Load data from file
train_X, train_Y = load_date_from_file(train_filename)
validation_X, validation_Y = load_date_from_file(validation_filename)

# Shuffle data
train_X, train_Y = shuffle_arrays(np.array(train_X), np.array(train_Y))
validation_X, validation_Y = shuffle_arrays(np.array(validation_X), np.array(validation_Y))

# Initialize the network
nn = NeuralNetwork([16, 32, 10, 1])

# Train the network on the train set
nn.train(train_X, train_Y, epochs=10, learning_rate=0.01)

# Write weights to file
pickle.dump(nn.weights, open("w0", "wb"))
