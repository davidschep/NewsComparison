import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")


class TFIDF:
    
    def __init__(self, df):
        self.df = df
        try:
            nltk.download('stopwords') # Download necessary NLTK resources to remove stop words
        except nltk.exceptions.AlreadyDownloaded:
            # Handle the case where the resource is already downloaded
            pass
    def preprocess_articles(self, text):
        text = text.lower() # Convert data to lover case to remove multiple occurences of the same word
        text = re.sub(r'[^a-z\s]', '', text) # Remove special characters, digits and white space using regex expression
        stop_words = set(stopwords.words('english')) # Import stop words from nltk library
        words = [word for word in text.split() if word not in stop_words] # Extract all words that are not stop words
        stemmer = PorterStemmer() # Apply stemming using nltk PorterStemmer
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    # Create vocabulary of words from news articles
    def vocabulary(self, docs):
        vocab = set()
        for doc in docs:
            for word in doc:
                vocab.add(word)
        return sorted(vocab)

    # Calculate Term frequency
    def term_frequency(self, docs, vocab):
        tf = pd.DataFrame(0, index=range(len(docs)), columns=vocab)
        for i, document in enumerate(docs):
            num_words_doc = len(document)
            for word in document:
                tf.at[i, word] += document.count(word)/num_words_doc
        return tf

    # Inverse Document Frequency
    def inverse_document_frequency(self, docs, vocab):
        # We want to reduce the weight of terms that appear frequently in our collection of articles.
        idf = pd.Series(0, index=vocab) # Create series and set all elements to 0
        for word in vocab:
            counter = 0
            for doc in docs:
                if word in doc:
                    counter +=1
            idf[word] = np.log((len(docs))/(counter+1)) #TFIDF as stated in the slides of week 1
        return idf
    
    # TF-IDF
    def tf_idf(self, tf, idf, docs, vocab):
        tfidf = pd.DataFrame(index=range(len(docs)), columns=vocab)  # Create an empty DataFrame
        for i in range(len(docs)):
            tfidf.iloc[i] = tf.iloc[i]*idf  # Multiply TF values by IDF for each term
        tfidf = normalize(tfidf, norm='l2', axis=1)  # L2 normalization to scale vectors
        tfidf = pd.DataFrame(tfidf, columns=vocab)
        return tfidf

    def fit(self):
        # Apply preprocesing to news articles
        self.df['preprocessed_content'] = self.df['content'].apply(self.preprocess_articles)
        # Extract words from each news articles
        docs = self.df['preprocessed_content'].str.split()
        # Create a vocabulary corresponding to all the words in every news article
        vocab = self.vocabulary(docs)
        # Calculate the TF
        tf = self.term_frequency(docs, vocab)
        # Calculate the IDF
        idf = self.inverse_document_frequency(docs, vocab)
        # Multiply TF with IDF and calculate TF-IDF
        tfidf = self.tf_idf(tf, idf, docs, vocab)
        return tfidf