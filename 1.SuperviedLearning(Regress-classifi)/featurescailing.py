import numpy as np 
def zscore_normalize_feature(X):
    #X(m,n):input data ,m example ,n feature
   mu=np.mean(X,axis=0) #mu has(n,) axis0=operation trên m ,kqua còn n 
   sigma=np.std(X,axis=0) #sigma has (n,)
   #elecment wise
   X_norm=(X-mu)/sigma
   return X_norm
if __name__ == '__main__':
    X=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print(X)
    print(zscore_normalize_feature(X))
