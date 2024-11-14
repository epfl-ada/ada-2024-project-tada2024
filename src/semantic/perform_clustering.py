import pickle
import pandas as pd

from src.semantic.utils.clustering_methods import (
    Kmeans_Raw,
    Kmedoids_Man,
    Kmedoids_Cos,
    Spectral_NN
)

def read_embeddings(file_path):

    with open(file_path, 'rb') as file:
        embedding_file = pickle.load(file)
        concepts = list(embedding_file.keys())
        embeddings = list(embedding_file.values())
    return concepts, embeddings 

def calculate_n_clusters(file_path):
    # Read the category datasetconda create -n new_env python=3.8
    category_df = pd.read_csv(file_path, sep='\t', skiprows=12,header=None)
    category_df.columns = ['concept','category']

    # Collect all primary categories
    category_df['primary_category'] = category_df['category'].apply(lambda x: x.split('.')[1])

    # Calculate the number of clustering
    n_clusters = len(category_df['primary_category'].unique())
    return n_clusters

def run_all_clustering(embedding_file, category_file, state=520):
    # Read concepts and embedding values from the embedding file
    concepts, embeddings = read_embeddings(embedding_file)
    
    # Calculate the number of clusters based on the primary categories in the category file
    n_clusters = calculate_n_clusters(category_file)

    # Store the results in a dictionary
    results = {}

    # Run each clustering method and transfer the result into dict
    Kmeans_Raw_Clustering = Kmeans_Raw(embeddings, n_clusters, state)
    Kmeans_Raw_result = pd.DataFrame({'concept':concepts,'clustering':Kmeans_Raw_Clustering})
    results['K-Means'] = Kmeans_Raw_result
    
    Kmedoids_Man_Clustering = Kmedoids_Man(embeddings, n_clusters, state)
    Kmedoids_Man_result = pd.DataFrame({'concept':concepts,'clustering':Kmedoids_Man_Clustering})
    results['K-Medoids with Manhattan'] = Kmedoids_Man_result

    Kmedoids_Cos_Clustering = Kmedoids_Cos(embeddings, n_clusters, state)
    Kmedoids_Cos_result = pd.DataFrame({'concept':concepts,'clustering':Kmedoids_Cos_Clustering})
    results['K-Medoids with Cosine'] = Kmedoids_Cos_result

    Spectral_NN_Clustering = Spectral_NN(embeddings, n_clusters, state)
    Spectral_NN_result = pd.DataFrame({'concept':concepts,'clustering':Spectral_NN_Clustering})
    results['Spectral Clustering with NN'] = Spectral_NN_result
    
    return results

# Main block to execute when the script is run directly
if __name__ == "__main__":

    
    # Define file paths 
    embedding_file = ''    # Path to the embedding data file
    category_file = ''      # Path to the category data file

    # Run clustering on all models and print the results
    clustering_results = run_all_clustering(embedding_file, category_file)
    for model_name, clusters in clustering_results.items():
        print(f"\nModel: {model_name}")
        print(len(clusters))