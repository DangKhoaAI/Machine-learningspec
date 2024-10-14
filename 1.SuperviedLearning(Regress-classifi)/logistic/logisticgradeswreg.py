import numpy as np
import matplotlib.pyplot as plt
#? file này tính gradient của hàm logistic loss có regularization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#> hàm logistic reg
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            
    dj_db = 0.0                                       
    #> tính loss logistic
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                  
    dj_db = dj_db/m                                   
    #> thêm regularization
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  
#% test hàm
if __name__ == '__main__':
    np.random.seed(1)
    X_tmp = np.random.rand(5,3)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )