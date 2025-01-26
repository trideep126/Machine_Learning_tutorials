import numpy as np 

def momentum_gd(X,y,learning_rate,epochs,momentum=0.9):
    m,n = X.shape 
    theta = np.zeros(n)
    velocity = np.zeros(n)

    for _ in range(epochs):
        pred = np.dot(X,theta)
        grad = (1/m) * np.dot(X.T,(pred - y))

        velocity = momentum*velocity - learning_rate*grad
        theta += velocity 

    return theta 