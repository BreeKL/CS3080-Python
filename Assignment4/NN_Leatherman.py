import numpy as np
import pandas as pd
import os

# use myenv

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(y):
    exp_y = np.exp(y - np.max(y))
    return (exp_y / exp_y.sum(axis=1, keepdims=True))

def predict(X, weights, bias):
    return (np.dot(X,weights) + bias)

def predict_sigmoid(y):
    return (sigmoid(y) > 0.5).astype(int)

def predict_softmax(y):
    return (np.argmax(y, axis=1))

def init_weights(X,Y,mode):
    n_samples, n_features = X.shape

    if (mode == "logistic" or mode == "linear"):
        weights = np.zeros(n_features)
        bias = 0
    elif (mode == "categorical"):
        n_samples, n_classes = Y.shape
        weights = np.zeros((n_features, n_classes))
        bias = np.zeros(n_classes)
    elif (mode == "neural"):
        n_hidden = 5
        n_output = Y.shape[1]
        weightsL1 = np.random.rand(n_features, n_hidden) * 0.01
        biasL1 = np.zeros(n_hidden)
        weightsL2 = np.random.rand(n_hidden, n_output) * 0.01
        biasL2 = np.zeros(n_output)

        weights = [weightsL1, weightsL2]
        bias = [biasL1, biasL2]
    else:
        raise ValueError("Unsupported mode")
    
    return weights, bias
    
def forward_pass(X, weights, bias, mode):
    if (mode == "linear"):
        return predict(X, weights, bias)
    elif (mode == "logistic"):
        y = predict(X, weights, bias)
        return sigmoid(y)
    elif (mode == "categorical"):
        y = predict(X, weights, bias)
        return softmax(y)
    elif (mode == "neural"):
        y1 = predict(X, weights[0], bias[0])
        y1 = sigmoid(y1)
        y2 = predict(y1,weights[1],bias[1])
        return [softmax(y2), y1]
    else:
        raise ValueError("Unsupported mode")
    

def backward_pass(X,Y,y_pred,weights,bias,lr,mode):
    n_samples, n_features = X.shape

    if (mode == "linear" or mode == "logistic"):
        dw = np.dot(X.T, (y_pred - Y)) / n_samples
        db = np.mean(y_pred - Y)
    elif (mode == "categorical"):
        dw = np.dot(X.T, (y_pred - Y)) / n_samples
        db = np.mean(y_pred - Y, axis=0)
    elif (mode == "neural"):
        dL2 = y_pred[0] - Y
        dwL2 = np.dot(y_pred[1].T, dL2) # / n_samples
        dbL2 = np.mean(dL2, axis=0)
        dL1 = np.dot(dL2, weights[1].T) * (y_pred[1] * (1 - y_pred[1]))
        dwL1 = np.dot(X.T, dL1) # / n_samples
        dbL1 = np.mean(dL1, axis=0)
    
    if (mode == "neural"):
        weights[0] -= lr * dwL1
        bias[0] -= lr * dbL1
        weights[1] -= lr * dwL2
        bias[1] -= lr * dbL2
    else:
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

def show_loss(Y, y_pred, ep, mode):
    if mode == 'linear':
        mse = np.mean((y_pred - Y) ** 2)
        print(f"Epoch {ep}, MSE: {mse:.4f}")
    elif mode == 'logistic':
        # For logistic regression, we can use binary cross-entropy loss
        loss = -np.mean(
        Y * np.log(y_pred + 1e-15) + (1 - Y) * np.log(1 - y_pred + 1e-15))
        print(f"Epoch {ep}, Loss: {loss:.4f}")
    elif mode == 'categorical' or mode == 'neural':
        # For categorical cross-entropy loss
        loss = -np.mean(Y * np.log(y_pred[0] + 1e-15))
        print(f"Epoch {ep}, Loss: {loss:.4f}")

def train(X, Y, mode):
    lr = 0.1
    epochs = 401
    weights, bias = init_weights(X, Y, mode)

    for i in range(epochs): 
        y_pred = forward_pass(X, weights, bias, mode)
        weights, bias = backward_pass(X, Y, y_pred, weights, bias, lr, mode)

        if (i % 100 == 0): 
            show_loss(Y, y_pred, i, mode)

    return weights, bias

def prepare_data(target, mode):
    train_file = "Assignment4/train_energy_data.csv"
    test_file = "Assignment4/test_energy_data.csv"
    
    if not os.path.exists(train_file): 
        raise FileNotFoundError(f"{train_file} does not exist")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"{test_file} does not exist")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df['Residential'] = (train_df['Building Type'] == 'Residential').astype(int)
    train_df['Commercial'] = (train_df['Building Type'] == 'Commercial').astype(int)
    train_df['Industrial'] = (train_df['Building Type'] == 'Industrial').astype(int)

    test_df['Residential'] = (test_df['Building Type'] == 'Residential').astype(int)
    test_df['Commercial'] = (test_df['Building Type'] == 'Commercial').astype(int)
    test_df['Industrial'] = (test_df['Building Type'] == 'Industrial').astype(int)

    train_df['Weekend'] = (train_df['Day of Week'] == 'Weekend').astype(int)
    train_df['Weekday'] = (train_df['Day of Week'] == 'Weekday').astype(int)

    test_df['Weekend'] = (test_df['Day of Week'] == 'Weekend').astype(int)
    test_df['Weekday'] = (test_df['Day of Week'] == 'Weekday').astype(int)

    train_df = train_df.drop(columns=['Building Type','Day of Week'])
    test_df = test_df.drop(columns=['Building Type','Day of Week'])

    X_train_df = train_df.drop(columns=target)
    Y_train_df = train_df[target]

    X_test_df = test_df.drop(columns=target)
    Y_test_df = test_df[target]

    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()

    if mode == 'linear':
        Y_train = Y_train_df.to_numpy()
        Y_test = Y_test_df.to_numpy()
    else:
        Y_train = Y_train_df.to_numpy(dtype='int')
        Y_test = Y_test_df.to_numpy(dtype='int')

    # Feature scaling (optional but improves optimization stability)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    return X_train, Y_train, X_test, Y_test, train_df, test_df

def save_results(train_df, test_df, y_pred_train, y_pred_test, target, prefix):
    y_train_df = pd.DataFrame(y_pred_train, columns=['Predicted'])
    y_test_df = pd.DataFrame(y_pred_test, columns=['Predicted'])

    if isinstance(target, list):
        for i, item in enumerate(target):
            train_df['Predicted ' + item]  = (y_train_df['Predicted'] == i).astype(int)
            test_df['Predicted ' + item]  = (y_test_df['Predicted'] == i).astype(int)

            train_accuracy = np.mean((train_df['Predicted ' + item]  == train_df[item]).astype(int)) * 100
            print(f"Training Accuracy for {item}: {train_accuracy}%")

            test_accuracy = np.mean((test_df['Predicted ' + item]  == test_df[item]).astype(int)) * 100
            print(f"Test Accuracy for {item}: {test_accuracy}%")

    else:
        train_df['Predicted ' + target]  = y_pred_train
        test_df['Predicted ' + target]  = y_pred_test

        train_accuracy = np.mean((train_df['Predicted ' + target]  == train_df[target]).astype(int)) * 100
        if train_accuracy > 0: print(f"Training Accuracy for {target}: {train_accuracy}%")

        test_accuracy = np.mean((test_df['Predicted ' + target]  == test_df[target]).astype(int)) * 100
        if test_accuracy > 0: print(f"Testing Accuracy for {target}: {test_accuracy}%")

        train_df.to_csv(prefix+'train_results.csv', index=False)

        test_df.to_csv(prefix+'test_results.csv', index=False)

# demo linear regression
target = 'Energy Consumption'
X_train, Y_train, X_test, Y_test, train_df, test_df = prepare_data(target, mode='linear')
weights, bias = train(X_train, Y_train, mode='linear')
y_pred_train = predict(X_train, weights, bias)
y_pred_test = predict(X_test, weights, bias)
print("Training Predictions:", y_pred_train[:5])
print("Training Actual:", Y_train[:5])
print("Test Predictions:", y_pred_test[:5])
print("Test Actual:", Y_test[:5])
save_results(train_df, test_df, y_pred_train, y_pred_test, target, 'Linear_')

# demo logistic regression
target = 'Residential'
X_train, Y_train, X_test, Y_test, train_df, test_df = prepare_data(target, mode='logistic')
weights, bias = train(X_train, Y_train, mode='logistic')
y_pred_train = predict(X_train, weights, bias)
y_pred_train = predict_sigmoid(y_pred_train) # Apply sigmoid for binary classification
y_pred_test = predict(X_test, weights, bias)
y_pred_test = predict_sigmoid(y_pred_test) # Apply sigmoid for binary classification
print("Training Predictions:", y_pred_train[:5])
print("Training Actual:", Y_train[:5])
print("Test Predictions:", y_pred_test[:5])
print("Test Actual:", Y_test[:5])
save_results(train_df, test_df, y_pred_train, y_pred_test, target, 'Logistic_')

# demo categorical classification
target = ['Residential', 'Commercial', 'Industrial']
X_train, Y_train, X_test, Y_test, train_df, test_df = prepare_data(target, mode='categorical')
weights, bias = train(X_train, Y_train, mode='categorical')
y_pred_train = predict(X_train, weights, bias)
y_pred_train = predict_softmax(y_pred_train) # Apply softmax for multi-class classification
y_pred_test = predict(X_test, weights, bias)
y_pred_test = predict_softmax(y_pred_test) # Apply softmax for multi-class classification
print("Training Predictions:", y_pred_train[:5])
print("Training Actual:", Y_train[:5])
print("Test Predictions:", y_pred_test[:5])
print("Test Actual:", Y_test[:5])
save_results(train_df, test_df, y_pred_train, y_pred_test, target, 'Categorical_')

# demo neural network classification
target = ['Residential', 'Commercial', 'Industrial']
X_train, Y_train, X_test, Y_test, train_df, test_df = prepare_data(target, mode='neural')
weights, bias = train(X_train, Y_train, mode='neural')
y_pred_train = forward_pass(X_train, weights, bias, mode='neural')
y_pred_train = predict_softmax(y_pred_train[0]) # Apply softmax for multi-class classification
print("Training Predictions:", y_pred_train[:5])
print("Training Actual:", Y_train[:5])
y_pred_test = forward_pass(X_test, weights, bias, mode='neural')
y_pred_test = predict_softmax(y_pred_test[0]) # Apply softmax for multi-class classification
print("Test Predictions:", y_pred_test[:5])
print("Test Actual:", Y_test[:5])
save_results(train_df, test_df, y_pred_train, y_pred_test, target, 'Neural_')