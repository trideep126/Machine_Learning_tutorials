import numpy as np

class LinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.lr = learning_rate 
        self.n_iter = iterations 
        self.weights = None
        self.bias = None 

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X,self.weights) + self.bias 

            dw = (1/n_samples) * np.dot(X.T,(y - y_pred))
            db = (1/n_samples) * np.sum(y - y_pred)

            self.weights -= self.lr*dw 
            self.bias -= self.lr*db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
