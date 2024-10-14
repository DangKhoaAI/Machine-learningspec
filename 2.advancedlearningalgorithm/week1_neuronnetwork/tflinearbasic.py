import numpy as np
import tensorflow as tf
#? file này implement linear function với tensorflow  , so sánh nhẹ với numpy
#% tạo data sample
X_train = np.array([[1.0], [2.0]], dtype=np.float32)           
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       

#%tao layer linear
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
#>lúc này nếu gọi call w, b= linear_layer.get_weights() se khong tra ve gi w,b vi layer chua goi voi input data

#%call with input data
#> lúc này nếu call sẽ trả về w và b (nhưng là giá trị random)
a1 = linear_layer(X_train[0].reshape(1,1)) #>thay input thanh 2D (vi layer nhan du lieu 2D)
w, b= linear_layer.get_weights() #in ra
print(f"Random value cua tf w,b ={w,b}")

#%set value w va b qua set_weights 
set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b]) # set_weights takes a list of numpy arrays
w, b= linear_layer.get_weights()
print(f"Set value cua w,b tf ={w,b}")


#%thu so sanh tinh ket qua tf voi numpy
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b
print(f"Tensorflow prediction: {prediction_tf} \n Numpy Prediction: {prediction_np}  ")
