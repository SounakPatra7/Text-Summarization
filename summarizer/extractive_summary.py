import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer 
from nltk.cluster import KMeansClusterer
from scipy.spatial import distance_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Sentence Transformer model
model = SentenceTransformer('stsb-roberta-base')

def extractive_summarization(article, n_clusters=10, iterations=25):
    nltk.download('punkt')
    # Tokenize the article into sentences
    sentences = nltk.sent_tokenize(article)
    sentences = [sentence.strip() for sentence in sentences]

    # Create a DataFrame with the sentences
    df = pd.DataFrame(sentences, columns=['sentences'])

    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['sentences'])
    tfidf_scores = np.sum(tfidf_matrix, axis=1)

    # Normalize TF-IDF scores
    normalized_tfidf_scores = tfidf_scores / np.sum(tfidf_scores)
    df['tfidf_score'] = normalized_tfidf_scores

    # Get embeddings for each sentence
    df['embeddings'] = df['sentences'].apply(lambda sent: model.encode([sent])[0])

    # Convert embeddings into numpy array
    X = np.array(df['embeddings'].tolist())

    # Perform clustering
    kcluster = KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=iterations, avoid_empty_clusters=True)
    assigned_clusters = kcluster.cluster(X, assign_clusters=True)

    # Assign clusters and centroids to the DataFrame
    df['Cluster'] = assigned_clusters
    df['Centroid'] = df['Cluster'].apply(lambda x: kcluster.means()[x])

    # Calculate distance from centroid for each sentence
    df['distance_from_centroid'] = df.apply(lambda row: distance_matrix([row['embeddings']], [row['Centroid'].tolist()])[0][0], axis=1)

    # Sort sentences by combined score of distance from centroid and TF-IDF score
    df['combined_score'] = df['distance_from_centroid'] * df['tfidf_score']

    # Select top sentence from each cluster based on combined score
    sents = df.sort_values(by='combined_score', ascending=True).groupby('Cluster').head(1)['sentences'].tolist()

    # Create the final summary
    summary = ' '.join(sents)
    return summary
