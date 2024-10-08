import numpy as np 
import matplotlib.pyplot as plt
import random

def get_column(data, index):
    return [row[index] for row in data]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    # get tv (index = 0)
    tv_data = get_column(data, 0)

    # get radio (index = 0)
    radio_data = get_column(data, 1)

    # get newspaper (index = 0)
    newspaper_data = get_column(data, 2)

    # get sales (index = 0)
    sales_data = get_column(data, 3)

    # building X input and y output for training
    # Create list of features for in put
    X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y

def initialize_params():
    bias = 0
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)

    return [0 , -0.01268850433497871 , 0.004752496982185252 , 0.0073796171538643845]
    # return [bias, w1, w2, w3]

def predict(X_features, weights):
    result = X_features[0] * weights[0] + X_features[1] * weights[1] + X_features[2] * weights[2] + X_features[3] * weights[3]
    return result

def compute_loss(y_hat, y):
    return (y_hat - y) ** 2

def compute_gradient_w(X_features, y, y_hat):
    dl_dweights = [2 * (y_hat - y), 2 * X_features[1] * (y_hat - y), 2 * X_features[2] * (y_hat - y), 2 * X_features[3] * (y_hat - y)]
    return dl_dweights

def update_weight(weight, dl_dweight, lr):
    weights = [weight[0] - lr * dl_dweight[0], weight[1] - lr * dl_dweight[1], weight[2] - lr * dl_dweight[2], weight[3] - lr * dl_dweight[3]]
    return weights

def implement_linear_regression(X_feature, y_output, epoch_max = 50, lr = 1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_output)
    for epoch in range(epoch_max):
        print("epoch",epoch)
        for i in range(N):
            #get a sample - row i
            features_i = X_feature[i]
            y = y_output[i]

            y_hat = predict(features_i, weights)

            loss = compute_loss(y_hat, y)

            dl_dweights = compute_gradient_w(features_i, y, y_hat)

            weights = update_weight(weights, dl_dweights, lr)

            losses.append(loss)
    
    return weights, losses

X, y = prepare_data("d:/AIO/Code/advertising.csv")
W, L = implement_linear_regression(X, y, epoch_max=10000, lr=1e-5)
print(L[9999])
