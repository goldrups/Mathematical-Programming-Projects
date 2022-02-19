# nearest_neighbor.py
""""
Sam Goldrup
25 October 2021
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
import scipy.stats as ss
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    row_norm_diffs = la.norm(X-z, axis=1)  #get euclidean distance from rows
    mindex = np.argmin(row_norm_diffs)
    
    return X[mindex], min(row_norm_diffs) #return closest neighbor and the distance


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if isinstance(x, np.ndarray) == False:
            raise TypeError("it must be an array brO!") #fun way to call an error
        self.value = x #initialize the attributes
        self.left = None 
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        found_mom = False
        new_node = KDTNode(data)
        if self.root is None: #the tree is empty
            self.root = new_node
            self.k = np.shape(data)[0] #lets make k an integer
            self.root.pivot = 0
            found_mom = True
        if np.shape(data)[0] != self.k: 
            raise ValueError("length does not match", self.k)
        #if found_mom == False: #executes if there is already a root
        current = self.root
        while found_mom == False:
            if np.allclose(data, current.value) == True: #might not raise an error now
                raise ValueError("data already in tree")
            if new_node.value[current.pivot] < current.value[current.pivot] and current.left is None: # current > node AND it's time to insert
                found_mom = True
                current.left = new_node
                new_node.prev = current
                new_node.pivot = ((new_node.prev.pivot + 1) % self.k) #index for length of vector
            elif new_node.value[current.pivot] < current.value[current.pivot]: # current > node
                current = current.left
            if (new_node.value[current.pivot] > current.value[current.pivot] or np.allclose(new_node.value[current.pivot],current.value[current.pivot])) and current.right is None: # current <= node AND it's time to insert
                found_mom = True
                current.right = new_node
                new_node.prev = current
                new_node.pivot = ((new_node.prev.pivot + 1) % self.k)
            elif new_node.value[current.pivot] > current.value[current.pivot] or np.allclose(new_node.value[current.pivot],current.value[current.pivot]): # current <= node
                current = current.right
    

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, d_star):
            if current is None: #base case
                return nearest, d_star 
            x = current.value
            i = current.pivot
            if np.linalg.norm(x-z) < d_star: #current is closer to z than nearest
                nearest = current
                d_star = np.linalg.norm(x-z)
            if z[i] < x[i]: #search to the left
                nearest, d_star = KDSearch(current.left, nearest, d_star)
                if z[i] + d_star > x[i] or np.allclose(z[i] + d_star, x[i]): # search to the right if necessary
                    nearest, d_star = KDSearch(current.right, nearest, d_star)
            else: #search to the right
                nearest, d_star = KDSearch(current.right, nearest, d_star)
                if z[i] - d_star < x[i] or (np.allclose(z[i] - d_star, x[i])): #search to the left if needed
                    nearest, d_star = KDSearch(current.left, nearest, d_star)
            return nearest, d_star
        node, d_star_gold = KDSearch(self.root,self.root,np.linalg.norm(self.root.value-z))
        return node.value, d_star_gold

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors): #constructor
        self.n_neighbors = n_neighbors
    
    def fit(self, X,y): #initialize tree labels attributes
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        distances, indices = self.tree.query(z, k=self.n_neighbors) #find the nearest neighbors
        return ss.mode(self.labels[indices])[0][0] #get the most common neighbor
        

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float)  #training data
    y_train = data["y_train"]                   #training labels
    X_test = data["X_test"].astype(np.float)    #test data
    y_test = data["y_test"]                     #test labels

    handwriting_classifier = KNeighborsClassifier(n_neighbors) #create object
    handwriting_classifier.fit(X_train, y_train) #train the model
    handwriting_classifier.predict(X_test) #make the predictions

    y_output = np.array([handwriting_classifier.predict(vector) for vector in X_test]) #check for acccuracy

    return np.mean(np.equal(y_output, y_test))

        