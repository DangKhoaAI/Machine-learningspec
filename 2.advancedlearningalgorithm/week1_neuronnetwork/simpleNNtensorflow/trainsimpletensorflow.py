import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from loaddata import load_coffee_data
from sklearn.model_selection import train_test_split
#? file này tạo một NN nhỏ bằng tensorflow
#%load data
X,Y = load_coffee_data()
X=np.array(X) #(200,2)
Y=np.array(Y).reshape(-1,1) #tu (200,) thanh (200,1)
#%Normalize data
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
#%nhan ban du lieu
"""np.tile(array, reps) là hàm của NumPy, dùng để "lặp" một mảng theo một số lần nhất định"""
Xt = np.tile(Xn,(1100,1)) #200000,2
Yt= np.tile(Y,(1100,1))   #200000,1
print(Xt.shape, Yt.shape)   

#*tao sequential
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
"""
#cách lấy tham số của các layer trong sequential
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
"""
#%choose loss , optimizer
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

#%train model
model.fit(Xt,Yt,epochs=10)
#%display weigh sau khi fit
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

#*Lưu mô hình
model.save('my_model.h5')