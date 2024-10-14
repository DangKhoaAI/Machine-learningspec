import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from kerascustommodel import MyDenseLayer
#? file này load lại mô hình custom tensorflow và thực hiện predict
#*loadmodel
model = load_model('my_custom_model.h5', custom_objects={'CustomLayer': MyDenseLayer}) 
#*predict 
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
X_test = np.array([
    [200,0],  # positive example
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