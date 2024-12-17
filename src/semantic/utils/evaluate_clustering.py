import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist


def calculate_clustering_consistency(clustering_list):
    # Initialize lists to store ARI and NMI scores for each pairwise comparison
    ari_scores = []
    nmi_scores = []

    # Compute ARI and NMI for each pair of clustering results
    for i in range(len(clustering_list)):
        for j in range(i + 1, len(clustering_list)):
            ari_scores.append(
                adjusted_rand_score(clustering_list[i], clustering_list[j])
            )
            nmi_scores.append(
                normalized_mutual_info_score(clustering_list[i], clustering_list[j])
            )

    # Calculate the average ARI and NMI as overall consistency measures
    average_ari = np.mean(ari_scores)
    average_nmi = np.mean(nmi_scores)

    return average_ari, average_nmi


def get_primary_category(file_path):
    # Read the category dataset
    category_df = pd.read_csv(file_path, sep="\t", skiprows=12, header=None)
    category_df.columns = ["concept", "category"]

    # Collect all primary categories
    category_df["primary_category"] = category_df["category"].apply(
        lambda x: x.split(".")[1]
    )
    category_df = category_df[["concept", "primary_category"]]
    return category_df

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
    data = data.astype(np.float64)  # Ensure high precision
    cluster_centers = []
    intra_cluster_distances = []

    # Calculate cluster centers and intra-cluster distances
    for cluster_id in range(num_clusters):
        cluster_points = data[labels == cluster_id]
        if len(cluster_points) > 0:
            cluster_center = cluster_points.mean(axis=0)
            cluster_centers.append(cluster_center)
            intra_distances = cdist(cluster_points, [cluster_center])
            intra_cluster_distances.append(np.mean(intra_distances))
        else:
            # Handle empty clusters
            cluster_centers.append(np.zeros(data.shape[1]))
            intra_cluster_distances.append(0.0)

    cluster_centers = np.array(cluster_centers)

    # Check for empty clusters and ensure valid distances
    if len(cluster_centers) > 1:
        inter_cluster_distances = cdist(cluster_centers, cluster_centers)
        np.fill_diagonal(inter_cluster_distances, 0)
        avg_inter_cluster_distance = np.sum(inter_cluster_distances) / (num_clusters * (num_clusters - 1))
    else:
        avg_inter_cluster_distance = 0.0

    avg_intra_cluster_distance = np.mean(intra_cluster_distances)
    return avg_inter_cluster_distance, avg_intra_cluster_distance
