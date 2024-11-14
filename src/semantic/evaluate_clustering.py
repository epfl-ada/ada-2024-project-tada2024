import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def calculate_clustering_consistency(clustering_list):

    # Initialize lists to store ARI and NMI scores for each pairwise comparison
    ari_scores = []
    nmi_scores = []

    # Compute ARI and NMI for each pair of clustering results
    for i in range(len(clustering_list)):
        for j in range(i + 1, len(clustering_list)):
            ari_scores.append(adjusted_rand_score(clustering_list[i], clustering_list[j]))
            nmi_scores.append(normalized_mutual_info_score(clustering_list[i], clustering_list[j]))

    # Calculate the average ARI and NMI as overall consistency measures
    average_ari = np.mean(ari_scores)
    average_nmi = np.mean(nmi_scores)

    return average_ari, average_nmi


def get_primary_category(file_path):
    # Read the category dataset
    category_df = pd.read_csv(file_path, sep='\t', skiprows=12,header=None)
    category_df.columns = ['concept','category']

    # Collect all primary categories
    category_df['primary_category'] = category_df['category'].apply(lambda x: x.split('.')[1])
    category_df = category_df[['concept','primary_category']]
    return category_df


def map_clustering_category(category_df, clustering_df):
    # Merge the clustering results with primary categories
    merged_df = pd.merge(clustering_df, category_df, on='concept', how='inner')

    # Generate crosstab
    cross_tab = pd.crosstab(merged_df['clustering'], merged_df['primary_category'])
    
    # Convert the crosstab to long format and sort by count in descending order
    cross_tab_long = cross_tab.stack().reset_index()
    cross_tab_long.columns = ['clustering', 'primary_category', 'count']
    cross_tab_long = cross_tab_long.sort_values(by='count', ascending=False)

    # Initialize a dictionary to record the assigned relationships
    cluster_category_map = {}
    assigned_primary_categories = set()

    # Assign a unique primary_category to each Clustering
    for _, row in cross_tab_long.iterrows():
        clustering = row['clustering']
        primary_category = row['primary_category']
        
        # If the current primary_category has not been assigned to another Clustering
        if clustering not in cluster_category_map and primary_category not in assigned_primary_categories:
            cluster_category_map[clustering] = primary_category
            assigned_primary_categories.add(primary_category)

    # Calculate the accuracy for unique matches
    total_samples = len(merged_df)
    correct_matches = sum(merged_df['clustering'].map(cluster_category_map) == merged_df['primary_category'])
    accuracy = correct_matches / total_samples

    return cluster_category_map, accuracy