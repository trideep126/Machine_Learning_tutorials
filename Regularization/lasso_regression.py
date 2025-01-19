import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso 

np.random.seed(42)
X = np.linspace(0,10,100).reshape(-1,1)
y = 3*X + 2 + np.random.normal(0,2,(100,1))

alphas = [0,0.1,1,10]

for alpha in alphas:
  lasso = Lasso(alpha = alpha)
  lasso.fit(X,y)
  plt.plot(X,lasso.predict(X),label=f'Alpha:{alpha}')

plt.scatter(X,y,color='black',label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('L1 Regularization with different alpha values')
plt.legend()
plt.show()