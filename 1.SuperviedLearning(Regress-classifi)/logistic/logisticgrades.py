import numpy as np
import copy, math
import matplotlib.pyplot as plt
from logisticlossfunction import compute_cost_logistic
from logisticcomputegradient import compute_gradient_logistic
#? file này gradient descent logistic regresstion
#% khởi tạo data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#> hàm gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration (for graphing later)
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    #> lặp qua từng lần train
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history        
#% test hàm
if __name__ == "__main__":
    w_tmp  = np.zeros_like(X[0])
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w_out, b_out, _ = gradient_descent(X, y, w_tmp, b_tmp, alph, iters) 
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")