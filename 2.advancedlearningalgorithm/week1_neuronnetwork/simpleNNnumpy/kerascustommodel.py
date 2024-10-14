from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from loaddata import load_coffee_data
#? file này viết và train một custom Model tensorflow
#%dataprepare
X,Y = load_coffee_data() 
X=np.array(X) #(200,2)
Y=np.array(Y).reshape(-1,1) #tu (200,) thanh (200,1)
#%Normalize data
norm_l = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
#%nhan ban du lieu
"""np.tile(array, reps) là hàm của NumPy, dùng để "lặp" một mảng theo một số lần nhất định"""
Xt = np.tile(Xn,(1000,1)) #200000,2
Yt= np.tile(Y,(1000,1))   #200000,1
#% viết class custom layer
#> @ để có thể lưu (serialize)
@tf.keras.utils.register_keras_serializable() #
class MyDenseLayer(Layer):
    #> định nghĩa tham số ,các biến cần thiết của lớp
    def __init__(self, units,**kwargs):
        super().__init__(**kwargs)
        self.units = units
    #> hàm build tự động gọi khi kích thước đầu vào được xác định
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,name="w")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,name="b")
    #> hàm config sẽ config lại các tham số trong lớp cha
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units":self.units,
            }
        )
        return config
    #> hàm call sẽ thực hiện các phép tính toán custom
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.math.sigmoid(z)
#% viết class custom model
#> @ để có thể save model
@tf.keras.utils.register_keras_serializable()
class MySequentialModel(Model):
    #> hàm init: tạo các layer
    def __init__(self, units1, units2,**kwargs):
        super().__init__(**kwargs,name="custommodel")
        self.dense1 = MyDenseLayer(units1)
        self.dense2 = MyDenseLayer(units2)
    #> hàm call: thực hiện tính toán
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
if __name__ == "__main__":
    #%define inputs and model
    inputs = Input(shape=(2,))
    #model = MySequentialModel(3, 1)
    #% tạo model
    model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        MyDenseLayer(3, name = 'layer1'),
        MyDenseLayer(1, name = 'layer2')
     ]
)
    #%choose loss, optimizer
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    #*train model
    model.fit(
        Xt, Yt,
        epochs=10,
    )
    
    #*save model
    """
    Note: model subclass không thể lưu ở dạng h5
    (vì h5 lưu ở dạng đồ thị tĩnh và các model Sequential và Functional mới 
    biểu diễn được ở đồ thị tĩnh).
    Vì vậy có các cách sau:
    1. Lưu mô hình dưới định dạng TensorFlow SavedModel:
       (định dạng savedmodel có thể lưu subclassmodel)
       model.save('my_custom_model', save_format='tf')
    2. Lưu chỉ trọng số (weights) (vẫn ở dạng h5):
       Nếu bạn chỉ cần lưu lại trọng số mô hình và không cần lưu cấu trúc của nó,
       bạn có thể dùng phương pháp save_weights.
       model.save_weights('my_custom_weights.h5') 
       OR model.save_weights('my_custom_weights') 
    - Và khi cần nạp lại mô hình:
       model = MySequentialModel(3, 1)  # Khôi phục cấu trúc mô hình
       model.load_weights('my_custom_weights.h5')  # Nạp trọng số đã lưu
    3. chuyển về dạng sequential Layer keras (save dạng h5)   
        model.save('my_custom_model.h5')
    -và khi load model cần cho biết custom object
        loaded_model = load_model('my_custom_model.h5', custom_objects={'MyDenseLayer': MyDenseLayer})

    """
    model.summary()
    model.save('my_custom_model.h5')



