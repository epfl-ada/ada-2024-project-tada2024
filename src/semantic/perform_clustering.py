import argparse
import os
import pickle
import pandas as pd
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from utils.clustering_methods import Kmeans_Raw, Kmedoids_Man, Kmedoids_Cos, Spectral_NN

from src.semantic.utils.pca import reduce_with_pca



def read_embeddings(file_path):
    with open(file_path, "rb") as file:
        embedding_file = pickle.load(file)
        concepts = list(embedding_file.keys())
        embeddings = list(embedding_file.values())
    return concepts, embeddings


def calculate_n_clusters(file_path):
    # Read the category datasetconda
    category_df = pd.read_csv(file_path, sep="\t", skiprows=12, header=None)
    category_df.columns = ["concept", "category"]

    # Collect all primary categories
    category_df["primary_category"] = category_df["category"].apply(
        lambda x: x.split(".")[1]
    )

    # Calculate the number of clustering
    n_clusters = len(category_df["primary_category"].unique())
    print(f"{n_clusters} categories found in wikispeedia.")
    return n_clusters


def save_clusters(clusters, output_dir, file_name):
    # Create embedding output directory
    clustering_output_dir = os.path.join(output_dir)
    if not os.path.exists(clustering_output_dir):
        os.makedirs(clustering_output_dir)

    # Save pkl embeddings in output dir
    clustering_output_path = os.path.join(clustering_output_dir, file_name)
    with open(clustering_output_path, "wb") as file:
        pickle.dump(clusters, file)

    print("Saved clusters to: ", clustering_output_path)


def run_all_clustering(embedding_file, category_file, state=520):
    # Read concepts and embedding values from the embedding file
    concepts, embeddings = read_embeddings(embedding_file)

    # Calculate the number of clusters based on the primary categories in the category file
    n_clusters = calculate_n_clusters(category_file)


def perform_clusterings(embeddings, concepts, n_clusters, state):
    results = {}

    methods = {
        "K-Means": Kmeans_Raw,
        "K-Medoids_Manhattan": Kmedoids_Man,
        "K-Medoids_Cosine": Kmedoids_Cos,
        "Spectral_Clustering_NN": Spectral_NN,
    }

    # Run each clustering method and transfer the result into dict
    for method_name, method in tqdm(methods.items()):
        result = method(embeddings, n_clusters, state)
        results[method_name] = pd.DataFrame({"concept": concepts, "clustering": result})

    return results
  
def run_all_clustering(embedding_file, category_file, state=520):
    concepts, embeddings = read_embeddings(embedding_file)
    n_clusters = calculate_n_clusters(category_file)
    return perform_clusterings(embeddings, concepts, n_clusters, state)
    
def run_all_clustering_with_pca(embedding_file, category_file, state=520):
    concepts, embeddings = read_embeddings(embedding_file)
    n_clusters = calculate_n_clusters(category_file)
    new_embeddings = reduce_with_pca(embeddings, verbose=True)
    return perform_clusterings(new_embeddings, concepts, n_clusters, state)
    

def parse_argumnets():
    parser = argparse.ArgumentParser(description="Clustering script.")

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="all_mpnet_base_v2",
        help="The name of the embedding model (default: %(default)s)",
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="./data/semantic/output/embeddings",
        help="Directory to store embeddings (default: %(default)s)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/wikispeedia",
        help="Directory of wikispeedia dataset (default: %(default)s)",
    )

    parser.add_argument(
        "--clustering_dir",
        type=str,
        default="./data/semantic/output/clustering",
        help="Directory to store clustering results (default: %(default)s)",
    )
    return parser.parse_args()


def run_kimchi_cos(embedding_file, category_file, state=520):
    concepts, embeddings = read_embeddings(embedding_file)

    n_clusters = calculate_n_clusters(category_file)

    kmedoids = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=state)
    kmedoids.fit(embeddings)

    cluster_info = []

    for i in range(n_clusters):
        medoid_index = kmedoids.medoid_indices_[i]

        medoid_concept = concepts[medoid_index]
        medoid_embedding = embeddings[medoid_index]

        cluster_indices = [j for j, label in enumerate(kmedoids.labels_) if label == i]

        member_concepts = [concepts[j] for j in cluster_indices]
        member_embeddings = [embeddings[j] for j in cluster_indices]

        cluster_info.append(
            {
                "center_name": medoid_concept,
                "cluster_size": len(member_concepts),
                "member_names": member_concepts,
                "center_embed": medoid_embedding,
                "member_embeds": member_embeddings,
            }
        )

    df = pd.DataFrame(cluster_info)

    return df


if __name__ == "__main__":
    args = parse_argumnets()

    # Define file paths
    embedding_file = os.path.join(
        args.embeddings_path, f"{args.embedding_model_name}.pkl"
    )  # Path to the embedding data file
    category_file = os.path.join(
        args.dataset, "wikispeedia_paths-and-graph\categories.tsv"
    )  # Path to the category data file

    print("Reading embeddings from: ", embedding_file)
    print("Reading categories from: ", category_file)
    # Run clustering on all models and print the results
    clustering_results = run_all_clustering(embedding_file, category_file)
    for model_name, clusters in clustering_results.items():
        print(f"Model: {model_name}")
        print(len(clusters))

    save_clusters(
        clustering_results,
        args.clustering_dir,
        os.path.basename(embedding_file),
    )
