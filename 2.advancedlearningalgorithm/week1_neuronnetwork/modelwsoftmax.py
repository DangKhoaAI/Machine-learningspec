import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
#? file này cho 2 phiên bản model: 1. ko tối ưu numerical error 2. được tối ưu numerical error
def my_softmax(z):
    ez = np.exp(z)  #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

#% make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
#%model1 
model1 = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model1.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
#% model2 
# >model2 has softmax + loss combined(more stable and accurate)
model2 = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #>output is linear
    ]
)
model2.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    #>from_logit=True: loss function that the softmax operation should be included in the loss calculation
    optimizer=tf.keras.optimizers.Adam(0.001),
)

fitvar=int(input("Which model you want to fit 1.Model normal 2.ModelEnhance numerical error: "))
if fitvar==1:
    model1.fit(X_train,y_train,epochs=10)
    predict1 = model1.predict(X_train)
    print(f"two example output predict:\n {predict1[:2]}")
elif fitvar==2:
    model2.fit(X_train,y_train,epochs=10)
    predict2= model2.predict(X_train)
    print(f"two example output vectors:\n {predict2[:2]}")
    #>TO make prob predict:output model is linear, to predic prob need softmax convert it 
    sm_preferred = tf.nn.softmax(predict2)
    print(f"two example output prob:\n {sm_preferred[:2]}")
    #>TO select the most likely category,softmax is not required ,just find the index of the largest output using np.argmax().
    print("predict of first 5 elecment:")
    for i in range(5):
        print( f"{predict2[i]}, category: {np.argmax(predict2[i])}")

#>2 types of  loss funtion
''' 
Tensorflow has two potential formats for target values and the selection of the loss defines which is expected.
1. SparseCategorialCrossentropy: expects the target to be an integer corresponding to the index
 For example, if there are 10 potential target values, y would be between 0 and 9. 
2.CategoricalCrossEntropy: Expects the target value of an example to be one-hot encoded
 where the value at the target index is 1 while the other N-1 entries are zero. 
 An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].

'''