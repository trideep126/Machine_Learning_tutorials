import numpy as np

class MultipleLinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.lr = learning_rate 
        self.n_iter = iterations 
        self.weights = None
        self.bias = None
        self.feature_importances_ = None

    def _forward(self,X):
        return np.dot(X,self.weights) + self.bias 
    
    def _backward(self,X,y,y_pred):
        rows = X.shape[0]
        dw = (1/rows) * np.dot(X,(y - y_pred))
        db = (1/rows) * np.sum(y - y_pred)
        
        return dw,db

    def fit(self,X,y):
        rows,cols = X.shape
        self.weights = np.zeros(cols)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = self._forward(X)
            dw,db = self._backward(X,y,y_pred)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            self.feature_importances_ = np.abs(self.weights)/np.sum(np.abs(self.weights))

    def predict(self,X):
        return self._forward(X) 