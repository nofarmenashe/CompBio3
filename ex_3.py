import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return (1.0 - sigmoid(z)) * sigmoid(z)


def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))


def tanh_derivative(z):
    return 1 - np.power(np.tanh(z), 2)


class NeuralNetwork:
    def __init__(self, layers):
        self.activation_function = tanh
        self.activation_function_derivative = tanh_derivative
        # self.layers = len(net_arch)
        self.steps_per_epoch = 1000
        self.layers = layers
        self.weights = []

        # range of weight values (-1,1)
        for layer in range(len(self.layers) - 1):
            w = np.random.normal(0, 0.1, (self.layers[layer] + 1, self.layers[layer + 1]))
            # w = 2 * np.random.rand(self.layers[layer] + 1, self.layers[layer + 1]) - 1
            self.weights.append(w)

    #########
    # the fit function will train our network. In the last line,
    # `nn` represents the `NeuralNetwork` class and `predict` is the function in the NeuralNetwork class
    # that we will define later.
    #
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, data, labels, learning_rate=0.1, epochs=100, runs_per_epoch=15000):

        # Add bias units to the input layer -
        # add a "1" to the input data (the always-on bias neuron)
        ones = np.ones((1, data.shape[0]))
        Z = np.concatenate((ones.T, data), axis=1)

        for i in range(epochs * runs_per_epoch):
            print "iteration: ", i
            # Shuffle, Inner loop on each x
            # data, labels = shuffle_arrays(data, labels)

            # for x, label in data, labels:
            # We will now go ahead and set up our feed-forward propagation:
            sample = np.random.randint(data.shape[0])
            y = [Z[sample]]
            for i in range(len(self.weights) - 1):
                activation = np.dot(y[i], self.weights[i])
                activity = self.activation_function(activation)

                # add the bias for the next layer
                activity = np.concatenate((np.ones(1), np.array(activity)))
                y.append(activity)

            # last layer
            activation = np.dot(y[-1], self.weights[-1])
            activity = self.activation_function(activation)
            y.append(activity)

            # Now we do our back-propagation of the error to adjust the weights:
            error = labels[sample] - y[-1]
            delta_vec = [error * self.activation_function_derivative(y[-1])]

            # we need to begin from the back, from the next to last layer
            for i in range(len(self.layers) - 2, 0, -1):
                error = delta_vec[-1].dot(self.weights[i][1:].T)
                error = error * self.activation_function_derivative(y[i][1:])
                delta_vec.append(error)

            # Now we need to set the values from back to front
            delta_vec.reverse()

            # Finally, we adjust the weights, using the backpropagation rules
            for i in range(len(self.weights)):
                layer = y[i].reshape(1, self.layers[i] + 1)
                delta = delta_vec[i].reshape(1, self.layers[i + 1])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    def predict_single_data(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activation_function(np.dot(val, self.weights[i]))
            val = np.concatenate((np.ones(1).T, np.array(val)))
        return val[1]

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # x:      the input data array
    #########
    def predict(self, X):
        Y = None
        for x in X:
            y = np.array([[self.predict_single_data(x)]])
            if Y == None:
                Y = y
            else:
                Y = np.vstack((Y, y))
        return Y


def load_date_from_file(file_name):
    with open(file_name, mode='rb') as file:
        data = []
        for line in file:
            data.append(line.split())
    data = np.array(data)

    X = [item[0] for item in data]
    y = [int(item[1]) for item in data]

    # X = [list(x) for x in X]
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
        prediction = nn.predict_single_data(x)
        prediction = get_prediction_class(prediction)
        if y == prediction:
            success += 1

    return float(success) / len(validation_y) * 100


# Initialize the NeuralNetwork with
nn = NeuralNetwork([16, 32, 10, 1])

X, y = load_date_from_file("nn1.txt")
X = np.array(X)
y = np.array(y)

train_x, train_y = shuffle_arrays(X, y)

validation_x, validation_y = train_x[-5000:], train_y[-5000:]
train_x, train_y = train_x[:15000], train_y[:15000]


# Call the fit function and train the network for a chosen number of epochs
nn.fit(train_x, train_y, epochs=10, learning_rate=0.01)

# Show the prediction results
print("Final prediction")
success = calculate_success(validation_x, validation_y)
print "Success: ", success, "%"
