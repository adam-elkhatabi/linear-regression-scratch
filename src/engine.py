import numpy as np
import matplotlib.pyplot as plt

g = np.random.default_rng(101)

class LinearRegression:
    
    def __init__(self, epochs, learning_rate):
        self.w = None
        self.b = None
        self.epochs = epochs
        self.alpha = learning_rate
        self.w_history = []
        self.b_history = []
        self.losses = []

    
    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros((n_features, 1))
        self.b = 0

        for epoch in range(self.epochs):
            
            # prediction
            y_hat = X @ self.w + self.b

            # error
            error= y_hat - y

            # loss (MSE)
            L = (1 / (2*n_samples)) * np.sum(error**2)

            # track loss
            self.losses.append(L)

            # gradient
            dw = (1 / n_samples) * (X.T @ error)
            
            db = (1 / n_samples) * np.sum(error)

            # update
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)

            if epoch % (self.epochs // 10) == 0:
                print(f"the loss is {L}")


    def predict(self, X):
        return X @ self.w + self.b
    

# Building The Training Set :

n_samples = 100
n_features = 2

w_true = np.array([[5],
                   [2]])
b_true = 3

X = g.random((n_samples, n_features))
noise = g.normal(size=(n_samples, 1))

y = X @ w_true + b_true + noise

model = LinearRegression(1000, 0.01)
model.fit(X, y)
