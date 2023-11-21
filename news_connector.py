###########################
#
# Connect/cluster different articles into specific events that occured, based on dataframe as input
#
###########################


import math
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import collections
from numpy import linalg
from collections import Counter
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import re
import os
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")


def filter_common_words(df, threshold=0.95):
    # Count the frequency of each word in the dataframe
    word_counts = Counter()
    for idx, row in df.iterrows():
        words = row['content'].split()
        word_counts.update(words)

    # Determine the threshold for word frequency
    N = len(df)
    threshold = threshold * N

    # Filter out words that exceed the threshold
    common_words = []
    for word, count in word_counts.items():
        if count <= threshold:
            common_words.append(word)

    # Apply the filter function to each row of the "content" column
    def filter_row(row):
        words = row.split()
        filtered_words = []
        for word in words:
            if word in common_words:
                filtered_words.append(word)
        return ' '.join(filtered_words)

    # return filtered content
    df['content'] = df['content'].apply(filter_row)
    return df

def remove_infrequent_words(df, min_frequency=5):
    # Combine all strings in the specified column into a single string
    text = ' '.join(df['content'])

    # Tokenize the combined string into words
    words = text.split()

    # Count the frequency of each word using Counter
    word_counts = Counter(words)

    # Identify words that occur less than min_frequency times
    infrequent_words = []
    for word, count in word_counts.items():
       if count < min_frequency:
        infrequent_words.append(word)

    # Function to remove infrequent words from a text
    def remove_infrequent(text):
        return ' '.join([word for word in text.split() if word not in infrequent_words])

    # Apply the remove_infrequent function to the specified column
    df['content'] = df['content'].apply(remove_infrequent)

    return df

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
    
    # Update dataframe (500 articles: 4 min)
    df = filter_common_words(df)
    
    # Update dataframe (500 articles: 7 min)
    df = remove_infrequent_words(df)
    
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

def Cluster_Articles(k, df):
    kmeans = KMeans(k, random_state=123)
    tfidf_matrix = extract_documents(df)
    kmeans.fit(tfidf_matrix)
    df['cluster'] = kmeans.labels_
    return df