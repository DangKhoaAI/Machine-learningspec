import numpy as np
import copy, math
import matplotlib.pyplot as plt
from logisticlossfunction import compute_cost_logistic
#? file này tính gradient của logistic regression (logistic cost-> gradient)
#% tạo data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
#* viết hàm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#> hàm tính gradient
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           
    dj_db = 0.
    #> loop qua tất cả example
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]   
        # tính dj_dw cho từng w                    
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j] 
        #tính dj_db     
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
        
    return dj_db, dj_dw  

#%test compute gradient
if __name__ == '__main__':
    w_tmp = np.array([2.,3.])
    b_tmp = 1.
    dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X, y, w_tmp, b_tmp)
    print(f"dj_db: {dj_db_tmp}" )
    print(f"dj_dw: {dj_dw_tmp.tolist()}" )
