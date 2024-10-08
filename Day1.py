import numpy as np 
import matplotlib.pyplot as plt
import random

def get_column(data, index):
    return [row[index] for row in data]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    N = len(data)

    # get tv ( index =0)
    tv_data = get_column(data , 0)

    # get radio ( index =1)
    radio_data = get_column(data , 1)

    # get newspaper ( index =2)
    newspaper_data = get_column(data , 2)

    # get sales ( index =3)
    sales_data = get_column ( data , 3)

    # building X input and y output for training
    X = [ tv_data , radio_data , newspaper_data ]
    y = sales_data
    
    return X , y

def initialize_params():
    w1, w2, w3, b = (0.016992259082509283 , 0.0070783670518262355 , -0.002307860847821344 , 0)
    
    return w1 , w2 , w3 , b

def predict(x1, x2, x3, w1, w2, w3, b):
    return x1 * w1 + x2 * w2 + x3 * w3 + b

def compute_loss_mse(y, y_hat):
    return (y_hat - y) ** 2

def compute_loss_mae(y, y_hat):
    return abs(y_hat - y)

def compute_gradient_wi(xi, y, y_hat):
    return 2 * xi * (y_hat - y)

def compute_gradient_b(y, y_hat):
    return 2 * (y_hat - y)

def update_weight_wi(wi, dl_dwi, lr):
    return wi - lr * dl_dwi

def update_weight_b(b, dl_db, lr):
    return b - lr * dl_db

def implement_linear_regression(X_data, y_data, epoch_max = 50, lr = 1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            loss = compute_loss_mse(y, y_hat)

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)
    
    return (w1, w2, w3, b, losses)


def implement_linear_regression_nsamples(X_data, y_data, epoch_max = 50, lr = 1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        
        loss_total = 0
        dw1_total = 0
        dw2_total = 0
        dw3_total = 0
        db_total = 0

        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            loss = compute_loss_mse(y, y_hat)

            loss_total += loss

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        w1 = update_weight_wi(w1, dw1_total / N, lr)
        w2 = update_weight_wi(w2, dw2_total / N, lr)
        w3 = update_weight_wi(w3, dw3_total / N, lr)
        b = update_weight_wi(b, db_total / N, lr)

        losses.append(loss_total/N)

    return (w1, w2, w3, b, losses)


# X , y = prepare_data("d:/AIO/Code/advertising.csv")
# list = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
# print(list)

# (w1, w2, w3, b, losses) = implement_linear_regression(X, y)
# plt.plot(losses[:100])
# plt.xlabel("#iteration")
# plt.ylabel("Loss")
# plt.show()

# print(w1, w2, w3)

# tv = 19.2
# radio = 35.9 
# newspaper = 51.3 

# X, y = prepare_data("d:/AIO/Code/advertising.csv")
# (w1, w2, w3, b, losses) = implement_linear_regression(X, y, epoch_max=50, lr=1e-5)
# sales = predict(tv, radio, newspaper, w1, w2, w3, b)
# print(f'predicted sales is {sales}')


X, y = prepare_data("d:/AIO/Code/advertising.csv")
# (w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5)
# print(w1, w2, w3)


(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5)

# print(losses)
plt.plot(losses)
plt.xlabel("#epoch")
plt.ylabel("MAE Loss")
plt.show()

print ( w1 , w2 , w3 )