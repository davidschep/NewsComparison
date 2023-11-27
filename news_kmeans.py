import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

class KMeansAlgorithm:

    def __init__(self, X, k, max_iterations=100):
        """_summary_

        Args:
            X (dataframe): tf-idf matrix
            k (k): number of clusters
            max_iterations (int, optional): _description_. Defaults to 100.
        """
        self.X = X.values
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.num_articles = X.shape[0]

    def init_random_centroids(self):
        # Initializing KMeans by choosing random centroids 
        idx = np.random.choice(self.num_articles, size=self.k, replace=False) # Extract random indices from dataframe
        centroids = self.X[idx] # Pick random rows from the dataframe
        return centroids

    def calculate_euclidean_distances(self):
        # Calculate distances from vector/row to centroid
        num_centroids = self.centroids.shape[0]
        distances = np.zeros((num_centroids, self.num_articles))

        for centroid_idx in range(num_centroids):
            for article_idx in range(self.num_articles):
                distances[centroid_idx, article_idx] = np.sqrt(np.sum((self.centroids[centroid_idx, :] - self.X[article_idx, :]) ** 2))
        return distances

    def update_centroids(self, labels):
        # Calculate the mean of each cluster as new centroid
        new_centroids = []
        for k in range(self.k):
            mean = self.X[labels == k].mean(axis=0)
            new_centroids.append(mean)
        new_centroids = np.array(new_centroids)
        return new_centroids

    def plot_clusters(self, labels, centroids, iteration):
        # We will use PCA to plots clusters as the TFIDF matrix has many dimensions
        unique_labels = np.unique(labels)
        pca = PCA(n_components=3)
        data_3d = pca.fit_transform(self.X)
        centroids_3d = pca.transform(centroids)
        # 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        color_map = cm.get_cmap('turbo', len(unique_labels))
        # Plot data
        for label in unique_labels:
            indices = labels == label # like [1, 2, 1, 3, 1]
            ax.scatter(data_3d[indices, 0], data_3d[indices, 1], data_3d[indices, 2], label=f'Cluster {label}', c=[color_map(label)])
        # Plot centroids
        ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='x', s=80, label='Centroids')
        ax.set_title(f'PCA 3D - iteration {iteration}')
        ax.set_xlabel('Principal component 1')
        ax.set_ylabel('Principal component 2')
        ax.set_zlabel('Principal component 3')
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        clear_output(wait=True)
        plt.show()

    def fit(self):
        print("Clustering: KMeans fit()")
        
        # Run the Kmeans algorithm using helper functions
        self.centroids = self.init_random_centroids() # Init centroids

        for i in range(self.max_iterations):
            distances = self.calculate_euclidean_distances() # calculate eucledian distances
            labels = np.argmin(distances, axis=0) # Assign to the cluster with the centroid that has the minimum distance to that point
            new_centroids = self.update_centroids(labels) # Calculated centroids based on mean of the points in that cluster

            if np.all(new_centroids == self.centroids): # If no new centroids break loop
                print("Clustering: Kmeans has converged!")
                break

            self.centroids = new_centroids
            #self.plot_clusters(labels, self.centroids, i) # Plot PCA 3D plot

        return labels