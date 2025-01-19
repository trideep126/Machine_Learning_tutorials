import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

np.random.seed(42)
X = np.linspace(0,10,100).reshape(-1,1)
y = 3*X + 2 + np.random.normal(0,2,(100,1))

ratios = [0,0.5,1]

for ratio in ratios:
  elastic_net = ElasticNet(alpha=1,l1_ratio = ratio)
  elastic_net.fit(X,y)
  plt.plot(X,elastic_net.predict(X),label=f'Alpha:{alpha}')

plt.scatter(X,y,color='grey',label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Elastic Net Regularization with different alpha values')
plt.legend()
plt.show()