###########################
#
# Connect/cluster different articles into specific events that occured, based on dataframe as input
#
###########################

import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import re
import warnings
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
warnings.filterwarnings("ignore")
from IPython.display import clear_output

# Function to preprocess content column
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(stemmed_tokens)

def vocabulary(docs):
    """Create vocabulary"""
    vocab = set()
    for doc in docs:
        for word in doc:
            vocab.add(word)
    return sorted(vocab)

def term_frequency(documents, vocabulary):
    tf_matrix = pd.DataFrame(0, index=np.arange(len(documents)), columns=vocabulary)
    for i, document in enumerate(documents):
        for word in document:
            tf_matrix.at[i, word] += 1
    return tf_matrix

def inverse_document_frequency(documents, vocabulary):
    idf = pd.Series(0, index=vocabulary)
    for word in vocabulary:
        counter = 0
        for document in documents:
            if word in document:
                counter += 1
        idf[word] = np.log((1+len(documents))/(counter+1))+1
    return idf

def tf_idf(tf_matrix, idf, documents):
    tfidf_matrix = tf_matrix.copy()
    for i in range(len(documents)):
        tfidf_matrix.iloc[i] = tf_matrix.iloc[i] * idf
    tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=vocab)
    return tfidf_matrix

def extract_documents(df):
    """
    Main TF-IDF Function
    Total time for 500 articles: 20 minutes

    Args:
        df (dataframe): dataframe with preprocessed content
        
    Returns:
        tf idf matrix
    """
    # Download NLTK resources
    nltk.download('stopwords')
    
    # Apply preprocessing to the 'content' column (500 articles: 5 sec)
    df['preprocessed_content'] = df['content'].apply(preprocess_text)
    
    # Following steps (500 articles: 7 min)
    # Extract documents
    docs = df['preprocessed_content'].str.split()
    # Vocabulary
    vocab = vocabulary(docs)
    # TF
    tf_matrix = term_frequency(docs, vocab)
    # IDF
    idf = inverse_document_frequency(docs, vocab)
    # TF-IDF
    tfidf_matrix = tf_idf(tf_matrix, idf, docs)
    return tfidf_matrix

class KMeansAlgorithm:

    def __init__(self, X, k, max_iterations=100):
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
      
      # Run the Kmeans algorithm using helper functions
      self.centroids = self.init_random_centroids() # Init centroids

      for i in range(self.max_iterations):
        distances = self.calculate_euclidean_distances() # calculate eucledian distances
        labels = np.argmin(distances, axis=0) # Assign to the cluster with the centroid that has the minimum distance to that point
        new_centroids = self.update_centroids(labels) # Calculated centroids based on mean of the points in that cluster

        if np.all(new_centroids == self.centroids): # If no new centroids break loop
          print("Kmeans has converged!")
          break

        self.centroids = new_centroids
        self.plot_clusters(labels, self.centroids, i) # Plot PCA 3D plot

      return labels

def KMeans(k, df):
    tfidf_matrix = extract_documents(df)
    Kmeans = KMeansAlgorithm(tfidf_matrix, k) # Optimal number of clusters is determined from the results of the elbow method
    labels = Kmeans.fit()
    df['cluster'] = labels
    return df

def elbow_method(data):
    # We will use elbow method to determine optimal number of clusters
    num_clusters = range(1, 30)
    wcss = []

    for k in num_clusters:
        # We will use Sklearn's Kmeans algorithm, but use random intialization as used in our implementation
        kmeans = KMeans(n_clusters=k, init='random', random_state=123)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method
    plt.plot(num_clusters, wcss, marker='o')
    plt.title('Elbow Method - KMeans with random initialization')
    plt.xlabel('Number of clusters')
    plt.ylabel('Wcss')
    plt.show()