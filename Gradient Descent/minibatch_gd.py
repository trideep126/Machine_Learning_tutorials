import numpy as np

def mini_batch_gd(X,y,learning_rate,iterations,batch_size):
    m,n = X.shape
    theta = np.zeros(n)

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0,m,batch_size):
            Xi = X_shuffled[i: i+batch_size]
            yi = y_shuffled[i: i+batch_size]
            gradient = np.dot(Xi.T,(np.dot(Xi,theta) - yi))
            theta -= learning_rate * gradient 

    return theta 

X = np.array([1,1],[1,2],[1,3],[1,4],[1,5])
y = np.array([1,2,3,4,5])
theta = mini_batch_gd(X,y,0.01,1000,2)
print(f"Optimized theta: {theta}") 