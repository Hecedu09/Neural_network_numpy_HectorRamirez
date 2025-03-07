import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

# Create datasets from scratch - Example for classification
def create_dataset(N=1000):
    """
    Generates a synthetic dataset with two classes using Gaussian distributions.

    Parameters:
    N (int): Number of samples to generate.

    Returns:
    X (numpy.ndarray): Feature matrix.
    Y (numpy.ndarray): Label vector with values 0 or 1.
    """
    gaussian_quantiles = make_gaussian_quantiles(
        mean=None,
        cov=0.1,
        n_samples=N,
        n_features=2,
        n_classes=2,
        shuffle=True,
        random_state=None
    )
    X, Y = gaussian_quantiles
    Y = Y[:, np.newaxis]  # Convert to column matrix
    return X, Y

# Activation functions
def sigmoid(x, derivate=False):
    """
    Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input to the activation function.
    derivate (bool): If True, calculates the derivative of the sigmoid function.

    Returns:
    numpy.ndarray: Sigmoid output or its derivative.
    """
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1)**2
    else:
        return 1 / (1 + np.exp(-x))

def relu(x, derivate=False):
    """
    ReLU activation function.

    Parameters:
    x (numpy.ndarray): Input to the activation function.
    derivate (bool): If True, calculates the derivative of the ReLU function.

    Returns:
    numpy.ndarray: ReLU output or its derivative.
    """
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)

# Loss function
def mse(y, y_hat, derivate=False):
    """
    Computes the Mean Squared Error (MSE).

    Parameters:
    y (numpy.ndarray): True values.
    y_hat (numpy.ndarray): Predictions.
    derivate (bool): If True, computes the derivative of the MSE.

    Returns:
    float or numpy.ndarray: Error value or its derivative.
    """
    if derivate:
        return (y_hat - y)
    else:
        return np.mean((y_hat - y)**2)

# Weight and bias initialization
def initialize_parameters_deep(layers_dims):
    """
    Initializes the weights and biases of the neural network.

    Parameters:
    layers_dims (list): List with the number of neurons in each layer.

    Returns:
    dict: Dictionary containing the network parameters (weights and biases).
    """
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1
        parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
    return parameters

# Neural network training
def train(x_data, y_data, learning_rate, params, training=True):
    """
    Forward and backward propagation for training the neural network.

    Parameters:
    x_data (numpy.ndarray): Input data.
    y_data (numpy.ndarray): True labels.
    learning_rate (float): Learning rate.
    params (dict): Dictionary with network parameters.
    training (bool): If True, adjusts weights and biases.

    Returns:
    numpy.ndarray: Neural network output.
    """
    params['A0'] = x_data

    # Forward propagation
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if training:
        # Backpropagation
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

        # Parameter update using gradient descent
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate

        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output

# Main function to train the model
def train_model():
    """
    Trains the neural network and visualizes the data.
    """
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]
    params = initialize_parameters_deep(layers_dims)
    error = []

    for _ in range(50000):
        output = train(X, Y, 0.001, params)
        if _ % 50 == 0:
            print(mse(Y, output))
            error.append(mse(Y, output))

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
