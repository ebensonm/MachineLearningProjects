"""
Random Forest Lab

Eric Manner
November 6, 2020
"""
import graphviz
import os
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

#python3 experimenter random_seed_job_maker -n-seeds 10 
# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""
    
    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]
    
    def match(self,sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        #compare the corresponding column to the question value
        return sample[self.column] >= self.value
        
    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))
    
def partition(data,question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    #use the given questions to return the splits
    mask_match = question.match(data.T)
    #now use the mask to select the data that matches
    left = data[mask_match]
    right = data[~mask_match]
    #now make sure they are not empty
    if (left.shape[0] == 0):
        left = None
    if (right.shape[0] == 0):
        right = None
    return left, right
    
#Problem 2    
def gini(data, column_class=-1):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""
    #get the number of samples and the number of class labels
    N,col = np.shape(data)
    #get the number of class values
    K, counts = np.unique(data[:,column_class], return_counts=True)
    #return the gini index
    return 1 - np.sum(np.square(counts/N))

def info_gain(left,right,G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""
    #get the shapes
    m_l,n_l = np.shape(left)
    m_r,n_r = np.shape(right)
    #calculate the ginis
    g_l = gini(left)
    g_r = gini(right)
    #now get return the value
    return G - (m_l/(m_l+m_r)*g_l + m_r/(m_l+m_r)*g_r)
    
# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False, column_class=-1):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""
    N, f_n = np.shape(data)
    n = f_n - 1
    root_n = int(np.ceil(np.sqrt(n)))
    if (column_class == -1):
        column_class = f_n - 1
    if (random_subset == True):
        list_features = np.random.choice(np.arange(0, f_n-1, 1), root_n).tolist()
        feature_names_search = np.array(feature_names)[list_features].tolist()
    else:
        feature_names_search = feature_names
    G = gini(data)
    best_question = None
    #initialize the value to optimize
    info_best = -np.inf
    #begin the optimization loop
    for it_f in range(len(feature_names)):
        if (column_class == it_f):
            continue
        #functionality for limiting features to split on
        if (feature_names[it_f] not in feature_names_search):
            continue
        #the list for unique vals
        val_list = list()
        for sample in range(N):
            #get the unique value to create the question with
            val = data[sample,it_f]
            if (val not in val_list):
                val_list.append(val)
                #create the question
                question = Question(column=it_f , value=val, feature_names=feature_names)
                left, right = partition(data, question)
                #make sure the partition counts exceed the necessary number
                if (right is not None and left is not None):
                    m_l,_ = np.shape(left)
                    m_r,_ = np.shape(right)
                    if (m_l >= min_samples_leaf and m_r >= min_samples_leaf):
                        #compute the info gain
                        gain = info_gain(left, right, G)
                        #now check if it is the best
                        if (gain > info_best):
                            info_best = gain
                            best_question = question
    return info_best, best_question
            
# Problem 4
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data,column_class=-1):
        #get the counts
        K, counts = np.unique(data[:,column_class], return_counts=True)
        #create the dictionary
        prediction = {K[i]: counts[i] for i in range(len(K))}
        #save as an attribute
        self.prediction = prediction

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, right_branch, left_branch):
        self.question = question
        self.right = right_branch
        self.left = left_branch

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)    
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False, column_class=-1):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    current_depth += 1
    info_best, best_question = find_best_split(data=data, feature_names=feature_names,       min_samples_leaf=min_samples_leaf, random_subset=random_subset, column_class=column_class)
    #if the tree is at max depth or meets other conditions
    if (current_depth == max_depth+1 or np.shape(data)[0] <= 2*min_samples_leaf or np.allclose(0, info_best)):
        decision_node = Leaf(data=data, column_class=column_class)
        return decision_node
    #add to the tree
    else:
        #partition the data
        left, right = partition(data, best_question)
        #recursive step
        left_node = build_tree(data=left, feature_names=feature_names, min_samples_leaf=min_samples_leaf, max_depth=max_depth, current_depth=current_depth, random_subset=random_subset, column_class=column_class)
        right_node = build_tree(data=right, feature_names=feature_names, min_samples_leaf=min_samples_leaf, max_depth=max_depth, current_depth=current_depth, random_subset=random_subset, column_class=column_class)
        #return a decision node
        decision_node = Decision_Node(best_question, right_node, left_node)
        return decision_node        
        
# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    if (isinstance(my_tree, Decision_Node)):
        question = my_tree.question
        value = int(question.match(sample))
        if (value == 1):
            new_tree = my_tree.left
        else:
            new_tree = my_tree.right
        return predict_tree(sample, new_tree)
    #check if we are on a leaf
    elif (isinstance(my_tree, Leaf)):
        #get the max class of the leaf
        prediction = my_tree.prediction
        return max(prediction, key=lambda k: prediction[k])
    else:
        raise ValueError("Found node of incorrect type: {}".format(type(my_tree)))
    
def analyze_tree(dataset,my_tree,column_class=-1):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    #get the relevant starting variables
    labels = dataset[:,column_class]
    N, class_num = np.shape(dataset)
    datapred = np.zeros(N)
    #now loop and get the predictions
    for i in range(N):
        prediction = predict_tree(dataset[i,:], my_tree)
        datapred[i] = prediction
    #now get the accuracy
    check_array = datapred == labels
    return np.sum(check_array)/(np.shape(check_array)[0])

# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    #intialize the forest scorer
    num_trees = len(forest)
    guess = np.zeros(num_trees)
    #loop through the forest
    for it, tree in enumerate(forest):
        pred = predict_tree(sample, tree)
        guess[it] = pred
    #now get the guess with the most counts
    K, counts = np.unique(guess, return_counts=True)
    max_arg = np.argmax(counts)
    return K[max_arg]
    
        
def analyze_forest(dataset,forest,column_class=-1):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    #get the relevant starting variables
    labels = dataset[:,column_class]
    N, class_num = np.shape(dataset)
    datapred = np.zeros(N)
    #now loop and get the predictions
    for i in range(N):
        prediction = predict_forest(dataset[i,:], forest)
        datapred[i] = prediction
    #now get the accuracy
    check_array = datapred == labels
    return np.sum(check_array)/(np.shape(check_array)[0])
    

# Problem 8
def prob8():
    """Use the file parkinsons.csv to analyze a 5 tree forest.
    
    Create a forest with 5 trees and train on 100 random samples from the dataset.
    Use 30 random samples to test using analyze_forest() and SkLearn's 
    RandomForestClassifier.
    
    Create a 5 tree forest using 80% of the dataset and analzye using 
    RandomForestClassifier.
    
    Return three tuples, one for each test.
    
    Each tuple should include the accuracy and time to run: (accuracy, running time) 
    """
    #load the data
    data_park = pd.read_csv('parkinsons.csv', index_col=0).to_numpy()
    m,n = np.shape(data_park)
    park_features = pd.read_csv('parkinsons_features.csv').columns[1:].tolist()
    #get the random samples from the data
    list_indeces = np.random.choice(np.arange(0,m-1,1), 130).tolist()
    sample = data_park[list_indeces]
    train_sample = sample[:100,:]
    test_sample = sample[100:,:]
    #shuffle the data for the third test
    np.random.shuffle(data_park)
    split_val = int(np.ceil(m*0.8))
    large_train = data_park[:split_val,:]
    large_test = data_park[split_val:,:]
    #create the list of trees
    forest = list()
    for i in range(5):
        #begin the timing
        start = time.time()
        #train forest
        forest.append(build_tree(sample, park_features, min_samples_leaf=15, random_subset=True))
    #analyze forest
    score = analyze_forest(test_sample, forest)
    end=time.time()
    #compute the total time
    total_time = np.round(end-start, 2)
    test_1 = (score, total_time)
    #now do the other test
    start = time.time()
    small_rf = RandomForestClassifier(n_estimators=5,min_samples_leaf=15)
    small_rf.fit(train_sample[:,:-1], train_sample[:,-1])
    y_hat = small_rf.predict(test_sample[:,:-1])
    #now get the accuracy
    check_array = y_hat == test_sample[:,-1]
    score = np.sum(check_array)/(np.shape(check_array)[0])
    end = time.time()
    total_time = np.round(end-start, 2)
    test_2 = (score, total_time)
    #now for the final test
    start = time.time()
    rf = RandomForestClassifier()
    rf.fit(large_train[:,:-1], large_train[:,-1])
    y_hat = rf.predict(large_test[:,:-1])
    #now get the accuracy
    check_array = y_hat == large_test[:,-1]
    score = np.sum(check_array)/(np.shape(check_array)[0])
    end = time.time()
    total_time = np.round(end-start,2)
    test_3 = (score, total_time)
    return test_1, test_2, test_3  
    
    
        
