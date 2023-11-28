###########################
#
# Connect/cluster different articles into specific events that occured, based on dataframe as input
#
###########################

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

import news_kmeans
import news_tfidf

def extract_documents(df):
    """
    Main TF-IDF Function
    Total time for 500 articles: 20 minutes

    Args:
        df (dataframe): dataframe with preprocessed content
        
    Returns:
        tf idf matrix (dataframe): size [n,n] where [i,j] corresponds to similarity article i and j
    """
    tfidf = news_tfidf.TFIDF(df)
    tfidf_matrix = tfidf.fit()

    return tfidf_matrix

def Cluster_Articles(k, data):
    """Main cluster articles function

    Args:
        k (int): nr clusters
        data (dataframe): PD dataframe with contents, etc.

    Returns:
        dataframe: df with ['clusters'] appended
    """
    tfidf_matrix = extract_documents(data)
    Kmeans = news_kmeans.KMeansAlgorithm(tfidf_matrix, k) # Optimal number of clusters is determined from the results of the elbow method
    labels = Kmeans.fit()
    data['cluster'] = labels
    return data

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