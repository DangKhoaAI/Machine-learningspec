import pandas as pd
import numpy as np
#? file này build một decision tree đơn giản 
#%tạo data mẫu
X_train = np.array(
[[1, 1, 1],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
#%build các hàm: tính entropy, chia data ,tính info gain , build tree recursive
#> tính entropy
def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    entropy = 0.
    if len(y)==0:
        entropy=0
        
    else:
        c=0
        for i in y:
            if i ==1:
                c+=1
        if c==0 or c==len(y): entropy=0
        else:
            p_1=c/(len(y))
            entropy=-p_1*np.log2(p_1)-(1-p_1)*np.log2(1-p_1)    
    return entropy
#> split dataset
def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i,feature]==1 :
            
            left_indices.append(i)
        else: right_indices.append(i)
        
    return left_indices, right_indices
#> tính infomation gain:
def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)    
    # variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    #calculate information gain
    information_gain = 0
    w_left=len(y_left)/len(y_node)
    w_right=len(y_right)/len(y_node)
    H_node=compute_entropy(y_node)
    H_left=compute_entropy(y_left)
    H_right=compute_entropy(y_right)
    information_gain=H_node-(w_left*H_left+w_right*H_right)
    return information_gain
#> hàm chọn best feature để split(feature khi split có infor gian cao nhất)
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    num_features = X.shape[1]
    best_feature = None
    if compute_entropy(y)==0:
        return -1
    else:
        best_feature=0
        for i in range(num_features):
            if compute_information_gain(X,y,node_indices,i)>compute_information_gain(X,y,node_indices,best_feature):
                best_feature=i
    return best_feature
#> build tree recursive
tree = []
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 
    #>Base case: Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
    #>chưa đến max depth: get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    # Split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    #>Recursive
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

#%tính thử entropy:
print("Entropy at root node: ", compute_entropy(y_train)) 
#%thử split dataset
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0
left_indices, right_indices = split_dataset(X_train, root_indices, feature)
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)
#%thử tính info gain:
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain ", info_gain0)
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain ", info_gain1)
info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain ", info_gain2)
#%thử best split
best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)
#%thử build recursive tree
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)