from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron


import numpy as np

def k_means(X, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

class LMS:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X, y, max_iterations=100):
        # Initialize weights randomly
        self.weights = np.random.randn(X.shape[1])
        
        for _ in range(max_iterations):
            for i in range(X.shape[0]):
                # Update weights based on prediction error
                error = y[i] - np.dot(X[i], self.weights)
                self.weights += self.learning_rate * error * X[i]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# K-means
kmeans_labels, kmeans_centroids = k_means(X, k=3)
sklearn_kmeans = KMeans(n_clusters=3)
sklearn_kmeans.fit(X)

# Perceptron
lms = LMS()
lms.fit(X, y)
sklearn_perceptron = Perceptron()
sklearn_perceptron.fit(X, y)

# Compare results
# print("K-means (custom) labels:")
# print(kmeans_labels[:10])
# print("K-means (scikit-learn) labels:")
# print(sklearn_kmeans.labels_[:10])

# print("\nK-means (custom) centroids:")
# print(kmeans_centroids)
# print("K-means (scikit-learn) centroids:")
# print(sklearn_kmeans.cluster_centers_)

# print("\nLMS (custom) predictions:")
# print(lms.predict(X[:10]))
# print("LMS (scikit-learn) predictions:")
# print(sklearn_perceptron.predict(X[:10]))

from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Load the Boston Housing dataset
boston = fetch_california_housing()
X = boston.data
y = boston.target

# K-means
kmeans_labels, kmeans_centroids = k_means(X, k=3)
sklearn_kmeans = KMeans(n_clusters=3)
sklearn_kmeans.fit(X)

# Linear Regression
lms = LMS()
lms.fit(X, y)
sklearn_lr = LinearRegression()
sklearn_lr.fit(X, y)

# Compare results
print("K-means (custom) labels:")
print(kmeans_labels[:10])
print("K-means (scikit-learn) labels:")
print(sklearn_kmeans.labels_[:10])

print("\nK-means (custom) centroids:")
print(kmeans_centroids)
print("K-means (scikit-learn) centroids:")
print(sklearn_kmeans.cluster_centers_)

print("\nLMS (custom) predictions:")
print(lms.predict(X[:10]))
print("LMS (scikit-learn) predictions:")
print(sklearn_lr.predict(X[:10]))