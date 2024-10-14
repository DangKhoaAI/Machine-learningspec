import numpy as np
import matplotlib.pyplot as plt
#? file này viết hàm cost function của logistic regression
#% khởi tạo data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) \
#% viết các hàm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#> hàm logistic loss
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
#% test hàm
if __name__ =="__main__":
  w_tmp = np.array([1,1])
  b_tmp = -3
  print(compute_cost_logistic(X, y, w_tmp, b_tmp))
  fig,ax = plt.subplots(1,1,figsize=(4,4))

  ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', s=100)
  ax.set_xlim(0, 4)
  ax.set_ylim(0, 3.5)
  ax.set_ylabel('$x_1$')
  ax.set_xlabel('$x_0$')
  plt.show