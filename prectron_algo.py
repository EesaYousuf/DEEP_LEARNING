from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
X, y = make_classification(n_samples=100, 
                           n_features=2, 
                           n_informative=1, 
                           n_redundant=0,
                           n_clusters_per_class=1, 
                           n_classes=2, 
                           random_state=41, 
                           hypercube=False, 
                           class_sep=10)

# Plotting the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Synthetic Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

