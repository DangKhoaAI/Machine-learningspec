import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
#? file này  implement linear regression bằng sklearn
#TODO : hoàn thành file này đi  
X=[[1,2,3,4],[5,6,7,8]]
scalar=StandardScaler() 
"""
scalar.fit()-> học ,tính toán tham số,nhưng k thay đổi dữ liệu
X_scale=transform(X)-> biến đổi dữ liệu dựa trên tham số đã học.
fit_transform->tính toán tham số , biến đổi dữ liệu 
(transform và fit_transform) thường dùng để biến đổi dữ liệu trước khi huấn luyện
"""
X_norm=scalar.fit_transform(X)
print(X_norm)