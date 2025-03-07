import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

def create_dataset(N=1000):
    """
    Generates a synthetic dataset with two classes using Gaussian distributions.

    Parameters:
    N (int): Number of samples to generate.

    Returns:
    X (numpy.ndarray): Feature matrix.
    Y (numpy.ndarray): Label vector with values 0 or 1.
    """
    X, Y = make_gaussian_quantiles(cov=0.1, n_samples=N, n_features=2, n_classes=2, shuffle=True)
    Y = Y[:, np.newaxis]  # Reshape Y to be a column vector
    
    print(X.shape)
    print(Y.shape)
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
    return X, Y

def sigmoid(x, derivate=False):
    """
    Computes the sigmoid activation function or its derivative.

    Parameters:
    x (numpy.ndarray): Input array.
    derivate (bool): If True, computes the derivative.

    Returns:
    numpy.ndarray: Sigmoid output or derivative.
    """
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1) ** 2
    return 1 / (1 + np.exp(-x))

def relu(x, derivate=False):
    """
    Computes the ReLU activation function or its derivative.

    Parameters:
    x (numpy.ndarray): Input array.
    derivate (bool): If True, computes the derivative.

    Returns:
    numpy.ndarray: ReLU output or derivative.
    """
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)

def mse(y, y_hat, derivate=False):
    """
    Computes the Mean Squared Error (MSE) or its derivative.

    Parameters:
    y (numpy.ndarray): True labels.
    y_hat (numpy.ndarray): Predicted labels.
    derivate (bool): If True, computes the derivative.

    Returns:
    float or numpy.ndarray: MSE loss or derivative.
    """
    if derivate:
        return (y_hat - y)
    return np.mean((y_hat - y) ** 2)

def initialize_parameters_deep(layers_dims):
    """
    Initializes weights and biases for a deep neural network.

    Parameters:
    layers_dims (list): List of layer sizes.

    Returns:
    dict: Dictionary containing initialized weights and biases.
    """
    parameters = {}
    L = len(layers_dims)
    for l in range(L - 1):
        parameters[f'W{l + 1}'] = np.random.rand(layers_dims[l], layers_dims[l + 1]) * 2 - 1
        parameters[f'b{l + 1}'] = np.random.rand(1, layers_dims[l + 1]) * 2 - 1
    return parameters

def train(x_data, y_data, params, learning_rate, training=True):
    """
    Performs forward and backward propagation for training a neural network.

    Parameters:
    x_data (numpy.ndarray): Input features.
    y_data (numpy.ndarray): True labels.
    params (dict): Dictionary containing network parameters.
    learning_rate (float): Learning rate for weight updates.
    training (bool): Whether to update weights.

    Returns:
    numpy.ndarray: Model output.
    """
    params['A0'] = x_data
    
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])
    
    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])
    
    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])
    
    output = params['A3']
    
    if training:
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])
        
        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])
        
        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])
        
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate
        
        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate
    
    return output

def train_model():
    """
    Trains a neural network using synthetic data and visualizes results.
    """
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]
    params = initialize_parameters_deep(layers_dims)
    errors = []
    
    for _ in range(50000):
        output = train(X, Y, params, learning_rate=0.001)
        if _ % 50 == 0:
            print(mse(Y, output))
            errors.append(mse(Y, output))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(errors)
    plt.title("Training Error Over Time")
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Error/Loss")
    
    data_test_x = (np.random.rand(1000, 2) * 2) - 1
    data_test_y = train(data_test_x, None, params, learning_rate=0.0001, training=False)
    
    y = np.where(data_test_y > 0.5, 1, 0)
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_test_x[:, 0], data_test_x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title("Test Data with Predicted Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.tight_layout()
    plt.show()
