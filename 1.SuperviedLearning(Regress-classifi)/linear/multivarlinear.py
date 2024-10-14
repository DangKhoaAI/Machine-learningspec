import copy, math
import numpy as np
import matplotlib.pyplot as plt
#? file này viết linear regression với đa biến (input nhiều feature)
#% tạo data
np.set_printoptions(precision=2)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
#* viết các hàm: tính cost , tính gradient , gradient descent
#> hàm tính cost
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0] #X.shape là (3,4) ,X.shape[0] là 3 (số hàng-số example)
    cost = 0.0
    #>loop qua từng example
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b #> phép dot X ,w sẽ tính tất cả feature song song
        cost = cost + (f_wb_i - y[i])**2       
    cost = cost / (2 * m)                          
    return cost
#> hàm tính gradient
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,)) #khởi tạo vecto n feature
    dj_db = 0.
    #>loop qua example
    for i in range(m):
        #error của 1 example
        err = (np.dot(X[i], w) + b) - y[i] 
        
        #loop qua feature , tính djdw của feature wj của vecto w(n feature)
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err*X[i,j]  
        #tính djdb (scalar không loop)
        dj_db = dj_db + err                      
    dj_dw = dj_dw / m                      
    dj_db = dj_db / m
    return dj_db, dj_dw
#> hàm gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
      Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration ( for graphing later)
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    #> loop qua từng lần traian
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(X, y, w, b))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
        
    return w, b, J_history
#%khởi tạo tham só
b_init = 0
w_init = np.zeros(4,)

#% set hyperparameter
iterations = 1000
alpha = 5.0e-7

#%run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init,compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
# in giá trị dự đoán
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

#% plot cost và iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
