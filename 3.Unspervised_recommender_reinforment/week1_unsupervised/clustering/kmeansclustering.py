import numpy as np
#? file này implement thuật toán Kmeans clustering 
#%load data
def load_data():
    X = np.load("data/ex7_X.npy")
    return X
X = load_data()
#% các hàm của Kmeans: runkmeans(giống gradient descent) , gán centroid, tính lại centroid , tự tạo centroid
#>> thuật toán kmeans
def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    #> loop qua từng iteration
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))        
        #% assign centroid
        idx = find_closest_centroids(X, centroids)
        #% compute new centroid
        centroids = compute_centroids(X, idx, K) 
    return centroids, idx
#>> hàm step1:gán dữ liệu vào centroid
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        l2=[]
        for j in range (K):
            l2.append(np.linalg.norm(X[i]-centroids[j] ,ord=2)**2)
        
        idx[i]=np.argmin(l2)
    return idx
#>> hàm step2: tính lại centroid
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    for j in range(K):
        assign=[]
        for i in range(m):
            if idx[i]==j:
                assign.append(X[i])
        centroids[j]=np.sum(assign,axis=0)/len(assign)
    return centroids
#>> hàm tự tạo centroid
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]    
    return centroids
#% test step1: gán vào centroid
initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx =find_closest_centroids(X, initial_centroids)
#% test step2: tính lại centroid
K = 3
centroids = compute_centroids(X, idx, K)
print("The centroids are:", centroids)
#% test Kmeans algorithm
K = 3
max_iters = 10
initial_centroids = kMeans_init_centroids(X, K)
centroids, idx = run_kMeans(X, initial_centroids, max_iters)

