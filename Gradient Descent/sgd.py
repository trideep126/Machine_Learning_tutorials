import numpy as np 

def sgd(X,y,learning_rate,iterations):
    m,n = X.shape
    theta = np.zeros(n)

    for _ in range(iterations):
        for i in range(m):
            rnd_idx = np.random.randint(m)
            xi = X[rnd_idx,rnd_idx+1]
            yi = y[rnd_idx,rnd_idx+1]
            gradient = np.dot(xi.T,(np.dot(xi,theta)-yi))
            theta -= learning_rate* gradient 

    return theta 

X = np.array([[1,1],[1,2],[1,3]])
y = np.array([1,2,3])
theta = sgd(X,y,0.01,100)
print(f"Optimized Theta:{theta}")
