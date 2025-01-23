import numpy as np 

def gradient_descent(X,y,learning_rate=0.01,epochs=100):
    n = len(X)
    w=b=0

    for _ in range(epochs):
        y_pred = np.dot(X,w) + b 
        
        dw = (2/n) * np.sum(np.dot(X.T,(y - y_pred)))
        db = (2/n) * np.sum(y - y_pred)

        w -= learning_rate * dw
        b -= learning_rate * db 

    return w,b 

X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])
w,b = gradient_descent(X,y)
print(f"Optimal Weights")