import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist



# Loads embedding data from the specified file.
def load_embedding_data(category_file, embedding_file):
    """
    Loads embedding data from the specified file.
    """
    # Load embeddings
    with open(embedding_file, "rb") as file:
        embedding = pickle.load(file)
        values = list(embedding.values())

    return embedding, values

# run clustering
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

def Kmedoids_Man(data, n_clusters, state=520):
    kmedoids_man = KMedoids(
        n_clusters=n_clusters, metric="manhattan", random_state=state, init="k-medoids++", 
    )
    kmedoids_man.fit(data)
    return kmedoids_man.labels_


def Kmedoids_Cos(data, n_clusters, state=520):
    kmedoids_cos = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=state,init="k-medoids++")
    kmedoids_cos.fit(data)
    return kmedoids_cos.labels_

def perform_clusterings(embeddings, concepts, n_clusters, state):
    results = {}

    methods = {
        "K-Medoids_Manhattan": Kmedoids_Man,
        "K-Medoids_Cosine": Kmedoids_Cos,
    }

    # Run each clustering method and transfer the result into dict
    for method_name, method in tqdm(methods.items()):
        result = method(embeddings, n_clusters, state)
        results[method_name] = pd.DataFrame({"concept": concepts, "clustering": result})

    return results
  
def run_all_clustering(embedding_file, category_file, state=520):

    # Read embeddings and concepts
    concepts, embeddings = read_embeddings(embedding_file)
    print(f"Loaded {len(concepts)} concepts and corresponding embeddings from {embedding_file}.")

    # Calculate the number of clusters
    n_clusters = calculate_n_clusters(category_file)
    print(f"Calculated {n_clusters} clusters based on {category_file}.")

    # Perform clustering
    clustering_results = perform_clusterings(embeddings, concepts, n_clusters, state)
    print(f"Perform {list(clustering_results.keys())} clustering algorithms successfully!")

    # Return the clustering results
    return clustering_results


def get_primary_category(file_path):
    # Read the category dataset
    category_df = pd.read_csv(file_path, sep="\t", skiprows=12, header=None)
    category_df.columns = ["concept", "category"]

    # Collect all primary categories
    category_df["primary_category"] = category_df["category"].apply(
        lambda x: x.split(".")[1]
    )
    category_df = category_df[["concept", "primary_category"]]
    
    print(category_df.head(5))
    return category_df




# evaluate embeddings
def calculate_purity(y_true, y_pred):
    # Create a confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)

    # For each cluster, take the maximum number of correctly assigned labels
    majority_sum = np.sum(np.max(contingency_matrix, axis=0))

    # Divide by the total number of samples to get the purity
    purity = majority_sum / np.sum(contingency_matrix)
    
    return purity

def calculate_entropy(labels, clusters, num_classes):
    clusters = np.array(clusters)
    labels = np.array(labels)
    total_samples = len(labels)
    total_entropy = 0

    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_labels = labels[cluster_indices]
        cluster_size = len(cluster_labels)
        
        # Calculate proportion of each class in the cluster
        proportions = np.array([np.sum(cluster_labels == cls) / cluster_size for cls in range(num_classes)])
        proportions = proportions[proportions > 0]  # Ignore zero proportions
        
        # Calculate entropy for the cluster
        cluster_entropy = -np.sum(proportions * np.log2(proportions))
        total_entropy += (cluster_size / total_samples) * cluster_entropy

    return total_entropy


def map_clustering_category(category_df, clustering_df):
    # Merge the clustering results with primary categories
    merged_df = pd.merge(clustering_df, category_df, on="concept", how="inner")

    # Generate crosstab
    cross_tab = pd.crosstab(merged_df["clustering"], merged_df["primary_category"])

    # Convert the crosstab to long format and sort by count in descending order
    cross_tab_long = cross_tab.stack().reset_index()
    cross_tab_long.columns = ["clustering", "primary_category", "count"]
    cross_tab_long = cross_tab_long.sort_values(by="count", ascending=False)

    # Initialize a dictionary to record the assigned relationships
    cluster_category_map = {}
    assigned_primary_categories = set()

    # Assign a unique primary_category to each Clustering
    for _, row in cross_tab_long.iterrows():
        clustering = row["clustering"]
        primary_category = row["primary_category"]

        # If the current primary_category has not been assigned to another Clustering
        if (
            clustering not in cluster_category_map
            and primary_category not in assigned_primary_categories
        ):
            cluster_category_map[clustering] = primary_category
            assigned_primary_categories.add(primary_category)

    # Calculate the accuracy for unique matches
    total_samples = len(merged_df)
    correct_matches = sum(
        merged_df["clustering"].map(cluster_category_map)
        == merged_df["primary_category"]
    )
    accuracy = correct_matches / total_samples

    # Calculate weighted F1 score for unique matches
    category_cluster_map = {value: key for key, value in cluster_category_map.items()}
    merged_df['category2cluster'] = merged_df['primary_category'].apply(lambda x: category_cluster_map[x])
    y_true = merged_df['category2cluster'].tolist()
    y_pred = merged_df['clustering'].tolist()

    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    purity_score = calculate_purity(y_true, y_pred)
    entropy = calculate_entropy(y_true, y_pred, 15)

    return cluster_category_map, merged_df, purity_score, entropy


def calculate_cluster_distances(data, labels, num_clusters):
    cluster_centers = []
    intra_cluster_distances = []

    # Calculate cluster centers and intra-cluster distances
    for cluster_id in range(num_clusters):
        cluster_points = data[labels == cluster_id]  # Extract points in the current cluster
        if len(cluster_points) > 0:
            cluster_center = cluster_points.mean(axis=0)  # Compute cluster center
            cluster_centers.append(cluster_center)

            # Compute distances within the cluster (point-to-center distances)
            intra_distances = cdist(cluster_points, [cluster_center])  # Pairwise distances
            intra_cluster_distances.append(np.mean(intra_distances))
        else:
            # Handle empty clusters if they exist
            cluster_centers.append(np.zeros(data.shape[1]))
            intra_cluster_distances.append(0.0)

    # Convert cluster centers to a NumPy array
    cluster_centers = np.array(cluster_centers)

    # Compute inter-cluster distances
    inter_cluster_distances = cdist(cluster_centers, cluster_centers)  # Pairwise distances between centers
    np.fill_diagonal(inter_cluster_distances, 0)  # Ignore self-distances

    # Average distances
    avg_inter_cluster_distance = np.sum(inter_cluster_distances) / (num_clusters * (num_clusters - 1))
    avg_intra_cluster_distance = np.mean(intra_cluster_distances)

    return avg_inter_cluster_distance, avg_intra_cluster_distance


def evaluate_embeddings(clustering_results, embedding_values, primary_category, model_name):
    """
    Evaluate the clustering results for a specific embedding model.
    """
    print(f"------Evaluation results for {model_name} embeddings:")

    # Initialize mapping and evaluation dictionaries
    mapping = {}
    evaluation = {}

    # Initialize a DataFrame to save evaluation results
    evaluation_df = pd.DataFrame(columns=['inter_distance', 'intra_distance', 'purity', 'entropy'])

    for key, value in clustering_results.items():

        # Calculate the purity and entropy
        cluster_category_mapping, merged_df, purity, entropy = map_clustering_category(primary_category, value)
        mapping[key] = cluster_category_mapping
        evaluation[key] = merged_df

        # Calculate the inter-cluster and intra-cluster distances
        data = np.array(embedding_values).astype(np.float64)
        labels = np.array(value['clustering'])
        avg_inter_dist, avg_intra_dist = calculate_cluster_distances(data, labels, 15)

        # Save the results to the DataFrame
        evaluation_df.loc[len(evaluation_df)] = [avg_inter_dist, avg_intra_dist, purity, entropy]

    # Set the index based on clustering methods
    evaluation_df.index = ['K-Medoids_Manhattan', 'K-Medoids_Cosine']
    print(evaluation_df)

    return evaluation_df, mapping, evaluation



def process_path_data(path_df, distance_df, output_path, if_save=True):
    """
    Process path data to split paths into steps and calculate distances between consecutive steps.
    """
    if not if_save:
		  # Get the maximum path_length as the total number of step columns
      max_steps = path_df['path_length'].max()

      # Define a function to split the path into steps
      def split_path_to_steps(row):
        steps = row['path'].split(';')  # Split the path string by ";"
        steps += [None] * (max_steps - len(steps))  # Pad with None to match the maximum length
        return steps

      # Apply the split function to each row in the 'path' column and create step columns
      steps_df = pd.DataFrame(
        path_df.apply(split_path_to_steps, axis=1).tolist(), 
        columns=[f'step{i+1}' for i in range(max_steps)]
      )

      # Combine the hashedIpAddress column with the step columns
      path_df_a = pd.concat([path_df['hashedIpAddress'], steps_df], axis=1)

      # Convert the distance dataset to a dictionary, ensuring symmetry
      distance_dict = {}
      for _, row in distance_df.iterrows():
        distance_dict[(row['Concept 1'], row['Concept 2'])] = row['Distance']
        distance_dict[(row['Concept 2'], row['Concept 1'])] = row['Distance']  # Add reverse pair for symmetry

      # Define a function to compute distances between consecutive steps in a path
      def compute_step_distances(row):
        steps = row.dropna().tolist()[1:]  # Extract steps, excluding the hashedIpAddress
        step_distances = []
        for i in range(len(steps) - 1):
            concept_pair = (steps[i], steps[i + 1])
            # Retrieve distance; if not found, assign NaN
            distance = distance_dict.get(concept_pair, np.nan)
            step_distances.append(distance)
        return step_distances

      # Apply the function to each row in the path DataFrame
      step_distances_list = path_df_a.apply(compute_step_distances, axis=1)

      # Find the maximum number of distances to define column names
      max_distances = max(len(distances) for distances in step_distances_list)
      distance_columns = [f'distance_{i+1}' for i in range(max_distances)]

      # Convert the list of step distances into a new DataFrame
      new_path_df = pd.DataFrame(step_distances_list.tolist(), columns=distance_columns)

      # Insert the hashedIpAddress column at the beginning of the new DataFrame
      new_path_df.insert(0, 'hashedIpAddress', path_df['hashedIpAddress'])

      # Save the resulting DataFrame to a CSV file
      new_path_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Data successfully saved to {output_path}")

    







