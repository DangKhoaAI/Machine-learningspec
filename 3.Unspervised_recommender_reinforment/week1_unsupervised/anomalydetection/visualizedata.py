import numpy as np
import matplotlib.pyplot as plt
from Files.utils import *
from loaddataanodec import *
from anomalydetection import *
#? file visualize anomaly detection (với gaussian estimation)
X_train, X_val, y_val = load_data()
#% visualize dataset
#>Create a scatter plot of the data
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 
plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()
#>plot the data with estimate gaussian mu và var
mu, var = estimate_gaussian(X_train) 
print(f'Mean : {mu},Variance : {var}')
p = multivariate_gaussian(X_train, mu, var) 
visualize_fit(X_train, mu, var)
plt.show()
#>plot data+ gaussian và khoanh điểm oulier
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)
outliers = p < epsilon
visualize_fit(X_train, mu, var)
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro', markersize= 10,markerfacecolor='none', markeredgewidth=2)
plt.show()