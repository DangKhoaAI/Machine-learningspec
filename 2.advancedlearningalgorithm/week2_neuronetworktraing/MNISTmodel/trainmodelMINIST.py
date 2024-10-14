import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from loaddata import load_data
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
#? file này xây dựng một neuron network đơn giản để dự đoán dữ liệu MNIST
#%prepare data
X,Y=load_data(filepath="MNISTdata.csv")
X=np.array(X)
Y=np.array(Y).reshape(-1,1)
Xt = np.tile(X,(100,1)) 
Yt= np.tile(Y,(100,1))
print(Xt.shape,Yt.shape)
#%create model
model=Sequential(
    [ 
    Input(shape=(400,)),
    Dense(units=25,activation="relu"),
    Dense (units=15,activation="relu"),
    Dense(units=10,activation="linear")
     
    ]
)
#% set loss, optimizer ,fit model với data
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=Adam(learning_rate=0.001))
model.fit(X,Y,epochs=50)
#% save model
model.save("MNISTmodel.h5")
