
CÁCH DÙNG CÁCH LỆNH SKLEARN 

-1.scalar.fit()-> học ,tính toán tham số,nhưng k thay đổi dữ liệu
    X_scale=transform(X)-> biến đổi dữ liệu dựa trên tham số đã học.
-2.fit_transform->tính toán tham số , biến đổi dữ liệu 
(transform và fit_transform) thường dùng để biến đổi dữ liệu trước khi huấn luyện
-3 predict(X): dự đoán nhãn hoặc giá trị mục tiêu (trong bài toán hồi quy) từ dữ liệu đầu vào X 
sau khi mô hình đã được huấn luyện
    model.fit(X_train,y_train)
prediction=model.predict(X_test)
-4 predict_proba(X):dự đoán xác suất của từng nhãn (trong bài toán phân loại)(predict_proba sẽ
trả về một mảng xác suất,chỉ dùng cho mô hình dự đoán xác suất)
    probabilities = model.predict_proba(X_test)
-5 score(X,y):điểm số của mô hình dựa trên dữ liệu X và nhãn thực tế y((accuracy) đối với bài toán phân loại 
hoặc hệ số xác định R2 đối với bài toán hồi quy)
    accuracy = model.score(X_test, y_test)
-6fit + predict ,học tham số và dự đoán ngay
    kmeans = KMeans(n_clusters=3)
l   abels = kmeans.fit_predict(X)
-7.inverse_transform:áp dụng ngược lại biến đổi (thường chỉ dùng với StandardScaler hoặc PCA)
    X_original = scaler.inverse_transform(X_scaled)
-8.get_params(): trả về siêu tham số (hyperparameter) mô hình 
    params = model.get_params()
-9.set_params(**params):Dùng để đặt hoặc thay đổi các siêu tham số của mô hình 
    model.set_params(n_estimators=100, max_depth=5)
-10. partial_fit():huấn luyện mô hình trên một phần nhỏ của dữ liệu trong nhiều lần lặp lại
(huấn luyện trên dữ liệu lớn)
    model.partial_fit(X_batch, y_batch)
-11.decision_function(X):Trả về các giá trị quyết định từ mô hình phân loại trước khi áp dụng hàm softmax hoặc sigmoid
(dùng trong các thuật toán như SVM) (Giá trị này có thể được dùng để tính xác suất hoặc phân loại tùy theo ngưỡng)
    decision_scores = model.decision_function(X_test)
-12cross_val_score(): Dùng để thực hiện đánh giá mô hình bằng cách chia dữ liệu thành nhiều tập con (k-fold cross-validation) và trả về điểm số hiệu suất trên mỗi tập.
    scores = cross_val_score(model, X, y, cv=5)


 **pineline: xâu chuỗi nhiều bước tiền xử lý và mô hình lại với nhau
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SomeClassifier())
])
pipeline.fit(X_train, y_train)
"""
