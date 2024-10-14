import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
#? file này load mô hình tensorflow đã lưu và thực hiện predict
#* Load mô hình đã lưu
model = load_model('my_model.h5')
#% xem thông số
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
#*predict 
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
#đưa từ xác suất sang 0/1
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
