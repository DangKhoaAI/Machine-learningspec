import numpy as np
import matplotlib.pyplot as plt
#? file này tính linear cost function  với regularization
np.set_printoptions(precision=8)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#> hàm cost function với regularization
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    #> tính root mean square error :cost
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   
        cost = cost + (f_wb_i - y[i])**2                                         
    cost = cost / (2 * m)                                              
    #> thêm L2 regularization: reg_cost
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          
    reg_cost = (lambda_/(2*m)) * reg_cost                              
    
    total_cost = cost + reg_cost                                       
    return total_cost   
#% thử hàm                                                
if __name__ == '__main__':
    np.random.seed(1)
    X_tmp = np.random.rand(5,6)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)