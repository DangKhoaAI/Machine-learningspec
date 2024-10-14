import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
#? file này implement logistic function với tensorflow  , so sánh nhẹ với numpy
#%tao du lieu train 
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
pos = Y_train == 1
neg = Y_train == 0

#%khoi tao model
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)
#model.summary() #shows the layers and number of parameters in the 
#% lấy layer từ model
logistic_layer = model.get_layer('L1')
#%set parameter cho layer
set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

#%thu so sanh predict tf va np
a_tf = model.predict(X_train[0].reshape(1,1))
print(f"Tensorflow predict: {a_tf}")

def sigmoidnp(x):
    return 1/(1+np.exp(-x))
a_np = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(f"Numpy predict: {a_np}")