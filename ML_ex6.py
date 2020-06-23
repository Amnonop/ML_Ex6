# Amnon Ophir 302445804, Ross Bolotin 310918610

import matplotlib.pyplot as plt
import numpy as np

# Globals
data_set = {}
accuracy_history = []
s_error_history = [None] * 2000

NN_1_ARCHITECTURE = [
    {"input_dim": 3, "output_dim": 3},
    {"input_dim": 3, "output_dim": 1}
]

NN_2_ARCHITECTURE = [
    {"input_dim": 3, "output_dim": 6},
    {"input_dim": 6, "output_dim": 1}
]


def init_layers(nn_architecture, seed=99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    num_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_vals = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_number = idx + 1

        # extracting the number of units in layers
        layer_inp_size = layer["input_dim"]
        layer_outp_size = layer["output_dim"]

        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_vals['W' + str(layer_number)] = np.random.randn(
            layer_outp_size, layer_inp_size)
        params_vals['b' + str(layer_number)] = np.random.randn(
            layer_outp_size, 1)
    return params_vals


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, z):
    sig = sigmoid(z)
    return dA * sig * (1 - sig)


def single_layer_fwd_prop(A_prev, W_curr, b_curr):
    # calculation of the input value for the activation function
    z_curr = np.dot(W_curr, A_prev) + b_curr

    # selection of activation function
    activation_func = sigmoid

    # return of calculated activation A and the intermediate Z matrix
    return activation_func(z_curr), z_curr


def full_fwd_prop(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    x_curr = X

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        x_prev = x_curr

        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        x_curr, Z_curr = single_layer_fwd_prop(x_prev, W_curr, b_curr)

        # saving calculated values in the memory
        memory["A" + str(idx)] = x_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return x_curr, memory


def get_cost_value(y_hat, y):
    # number of examples
    m = y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    s_error = np.linalg.norm(y_hat-y)/8
    return np.squeeze(cost)  , s_error


# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(y_hat, y):
    y_hat_ = convert_prob_into_class(y_hat)
    return (y_hat_ == y).all(axis=0).mean()


def single_layer_backward_propagation(dx_curr, w_curr, b_curr, z_curr, x_prev):
    # number of examples
    m = x_prev.shape[1]

    # selection of activation function
    backward_activation_func = sigmoid_backward

    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dx_curr, z_curr)

    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, x_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dx_prev = np.dot(w_curr.T, dZ_curr)

    return dx_prev, dW_curr, db_curr


def full_backward_propagation(y_hat, y, memory, params_values, nn_architecture):
    grads_values = {}

    # number of examples
    m = y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    y = y.reshape(y_hat.shape)

    # initiation of gradient descent algorithm
    dx_prev = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat));

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1

        dx_curr = dx_prev

        x_c = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dx_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dx_curr, W_curr, b_curr, Z_curr, x_c)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def train(x, y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture)
    # initiation of lists storing the history
    # of metrics calculated during the learning process
    cost_history = []

    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        y_hat, cashe = full_fwd_prop(x, params_values, nn_architecture)

        # calculating metrics and saving them in history
        cost, s_error = get_cost_value(y_hat, y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(y_hat, y)
        s_error_history[i] = s_error/100

        # step backward - calculating gradient
        grads_values = full_backward_propagation(y_hat, y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if i % 50 == 0:
            if verbose:
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if callback is not None:
                callback(i, params_values)

    return params_values , accuracy_history


def create_parity_vectors():
    data_set = {}
    for i in range(9):
        temp_str = np.binary_repr(i, width=3)
        temp_vec = np.array(list(temp_str))
        temp_par = np.bitwise_xor(temp_vec.item(0),temp_vec.item(1))
        temp_par = np.bitwise_xor(temp_par,temp_vec.item(2))
        data_set[temp_vec] = temp_par


def create_data_set():
    for i in range(8):
        temp_str = np.binary_repr(i, width=3)
        temp_vec = np.array(list(temp_str))
        temp_par = np.bitwise_xor(int(temp_vec.item(0)), int(temp_vec.item(1)))
        temp_par = np.bitwise_xor(int(temp_par), int(temp_vec.item(2)))
        data_set[temp_str] = int(temp_par)
    x_set = np.zeros((8, 3), dtype=int)
    y_set = np.zeros((8, 1), dtype=int)
    i = 0
    for key, value in data_set.items():
        for j in range(3):
            x_set[i, j] = int(key[j])
        y_set[i, :] = int(value)
        i = i + 1
    return x_set, y_set

    # SECTION_A
def section_one():
    x_set, y_set = create_data_set()
    for i in range(100):
        params_values, accuracy_history = train(np.transpose(x_set), np.transpose(y_set), NN_1_ARCHITECTURE, 2000, 2)  # etta=2
    plot1 = plt.figure(1)
    x = np.arange(1, 2001)
    plt.title("Mean Square Error per iteration, section A")
    plt.xlabel("iteration")
    plt.ylabel("Mean Square Error")
    plt.plot(x, s_error_history)


def section_two():
    x_set, y_set = create_data_set()
    for i in range(100):
        params_values, accuracy_history = train(np.transpose(x_set), np.transpose(y_set), NN_2_ARCHITECTURE, 2000, 2)  # etta=2
    plot2 = plt.figure(2)
    x = np.arange(1, 2001)
    plt.title("Mean Square Error per iteration, section B")
    plt.xlabel("iteration")
    plt.ylabel("Mean Square Error")
    plt.plot(x, s_error_history)
    plt.show()



def main():
    # SECTION_A
    section_one()
    # SECTION_B
    section_two()


if __name__ == "__main__":
    # execute only if run as a script
    main()
