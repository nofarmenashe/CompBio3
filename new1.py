import numpy as np
import pickle


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


def get_prediction_class(prediction):
    if prediction < 0.5:
        return 0
    return 1


def calculate_success(test_x, test_y):
    i = 0
    with open('results1', 'w') as predictions_file:
        for x, y in zip(test_x, test_y):
            prediction = nn.get_x_prediction(x)
            prediction = get_prediction_class(prediction)

            predictions_file.write(str(int(prediction)))
            if i != len(test_x) - 1:
                predictions_file.write("\n")

            i += 1


# Load data from file
X, y = load_date_from_file("nn1.txt")
X = np.array(X)
y = np.array(y)

# Load weights
data = pickle.load(open("w1", "rb"))

nn = NeuralNetwork([16, 32, 10, 1])

for i, w in enumerate(data):
    nn.weights[i] = w

# Run over the data and write the results to file "results0"
calculate_success(X, y)
