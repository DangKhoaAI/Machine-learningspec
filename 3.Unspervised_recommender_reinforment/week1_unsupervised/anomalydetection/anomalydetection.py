import numpy as np
from loaddataanodec import *
#? file implement thuật toán anomaly detection (với gaussian estimation)
#% load data
X_train, X_val, y_val = load_data()
X_train_high, X_val_high, y_val_high = load_data_multi()
#//print("The first 5 elements of X_train are:\n", X_train[:5]) 
#* các hàm anomaly detection:
#> hàm estimate gassian đơn biến (tính mu(vecto n feature) ,var(vecto n feature))
def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu=np.zeros(n)
    var=np.zeros(n)
    
    mu=np.mean(X,axis=0)
    var=np.var(X,axis=0)   
    return mu, var
#> hàm multivarite gaussian
def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """ 
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
#> hàm select threhold(duyệt qua các threhold và chọn threhold có F1score cao nhất)
def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):    
        predictions = (p_val < epsilon)
        tp = sum((predictions == 1) & (y_val == 1))
        prec=tp/np.sum(predictions)
        rec=tp/np.sum(y_val)
        F1=(2*prec*rec)/(prec+rec)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1
#% test hàm estimate gaussian
mu, var = estimate_gaussian(X_train) 
print(f'Mean : {mu},Variance : {var}')
#% test hàm select threhold
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val) 
#% test hàm trên dữ liệu nhiều chiều
#> Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)
#> Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
#> Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
#> Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))