import numpy as np 

def cost_function(X,y,theta):
    m = len(y)
    predictions = np.dot(X,theta)
    cost = (1/2*m) * np.sum((predictions - y)^2)
    return cost 

def gradient(X,y,theta):
    m = len(y)
    predictions = np.dot(X,theta) 
    grad = (1/m) * np.dot(X.T,(predictions - y))
    return grad 

def batch_gd(X,y,learning_rate=0.01,epochs=100):
    m,n = X.shape 
    theta = np.zeros(n)
    cost_history = []

    for _ in range(epochs):
        grad = gradient(X,y,theta)
        theta -= learning_rate * grad 
        cost = cost_function(X,y,theta)
        cost_history.append(cost)

    return theta,cost_history