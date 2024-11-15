 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def read_embeddings(file_path):

    with open(file_path, 'rb') as file:
        embedding_file = pickle.load(file)
        concepts = list(embedding_file.keys())
        embeddings = list(embedding_file.values())
    return concepts, embeddings

def reduce_with_pca(embeddings, pca_threshold=0.95, verbose=False):
    pca = PCA()
    pca_embeddings = pca.fit_transform(embeddings)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= pca_threshold) + 1
    reduced_embeddings = pca_embeddings[:, :n_components]
    
    if verbose :
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA: Cumulative Explained Variance Ratio')
        plt.show()
        print(f"Number of components explaining {pca_threshold*100}% of variance: {n_components}")

    return reduced_embeddings