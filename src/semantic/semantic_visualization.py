import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.spatial import ConvexHull

def plot_clustering_distribution(mpnet_values, mpnet_clustering, mpnet_mapping, d1, d2):
    """
    Generate a scatter plot for clustering distribution based on two embedding dimensions,
    with convex hulls representing cluster boundaries.
    """
    # Extract the specified dimensions
    mpnet_values_array = np.array(mpnet_values)[:, [d1, d2]]

    # Create a DataFrame for the selected dimensions and clustering
    mpnet_df = pd.DataFrame(mpnet_values_array, columns=["d1", "d2"])
    mpnet_df['clustering'] = mpnet_clustering['K-Medoids_Cosine']['clustering']

    # Add descriptive labels to clustering
    mpnet_df['clustering'] = mpnet_df['clustering'].apply(lambda x: f"{x}-{mpnet_mapping['K-Medoids_Cosine'][x]}")

    # Prepare the scatter plot
    plt.figure(figsize=(15, 10))
    scatter = sns.scatterplot(
        x='d1',
        y='d2',
        hue='clustering',
        palette='tab20',
        data=mpnet_df,
        s=20,
        alpha=0.7,
        edgecolor=None
    )

    # Function to draw convex hulls for each cluster
    def plot_convex_hull(points, color):
        if len(points) < 3:
            return  # A convex hull cannot be formed with fewer than 3 points
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.7)

    # Draw convex hulls for each cluster
    unique_clusters = mpnet_df['clustering'].unique()
    palette = sns.color_palette('tab20', len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_points = mpnet_df[mpnet_df['clustering'] == cluster][['d1', 'd2']].values
        plot_convex_hull(cluster_points, palette[i])

    # Add titles and labels
    plt.title('Clustering Distribution with Boundaries', fontsize=14)
    plt.xlabel('d1', fontsize=12)
    plt.ylabel('d2', fontsize=12)

    # Adjust legend and layout
    plt.legend(title='Clustering', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot
    plt.show()




def analyze_semantic_distances(mpnet_embedding, file_path):
    # Extract concept names and embeddings
    concept_names = list(mpnet_embedding.keys())
    embeddings = np.array(list(mpnet_embedding.values()))

    # Calculate cosine distances between all pairs of concepts
    results = []
    num_concepts = len(concept_names)
    for i in range(num_concepts):
        for j in range(i + 1, num_concepts):  # Avoid duplicate pairs
            concept_1 = concept_names[i]
            concept_2 = concept_names[j]
            distance = cosine(embeddings[i], embeddings[j])
            results.append((concept_1, concept_2, distance))

    # Convert results to a DataFrame
    distance_df = pd.DataFrame(results, columns=["Concept 1", "Concept 2", "Distance"])

    # Read the path data
    path_df = pd.read_csv(
        file_path,
        sep='\t',
        comment='#',
        header=None,
        names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]
    )

    # Handle missing values in the rating column
    path_df["rating"] = path_df["rating"].replace("NULL", pd.NA)

    # Calculate the length of each path
    path_df['path_length'] = path_df['path'].apply(lambda x: len(x.split(';')))

    # Count the number of paths for each length (filtering for lengths with > 2000 occurrences)
    length_counts = path_df['path_length'].value_counts().sort_index()
    length_counts = length_counts[length_counts > 2000]

    print("Path length counts:")
    print(length_counts)

    # Convert semantic distance data to a dictionary for fast lookups
    distance_dict = {}
    for _, row in distance_df.iterrows():
        distance_dict[(row['Concept 1'], row['Concept 2'])] = row['Distance']
        distance_dict[(row['Concept 2'], row['Concept 1'])] = row['Distance']  # Ensure symmetry

    # Function to compute semantic distances for a path
    def compute_path_distances(path, distance_dict, default_distance=0.5):
        concepts = path.split(';')
        distances = []
        for i in range(len(concepts) - 1):
            concept_pair = (concepts[i], concepts[i + 1])
            distance = distance_dict.get(concept_pair, default_distance)
            distances.append(distance)
        return distances

    # Process all paths and group them by path length
    path_lengths = []
    path_distances = []
    for path in path_df['path']:
        distances = compute_path_distances(path, distance_dict)
        path_lengths.append(len(distances))
        path_distances.append(distances)

    # Convert path lengths and distances into a DataFrame
    path_data = pd.DataFrame({'length': path_lengths, 'distances': path_distances})

    # Filter paths to include only specified lengths and sample 10 paths for each length
    filtered_lengths = [5, 7, 9]
    sampled_paths = (
        path_data[path_data['length'].isin(filtered_lengths)]
        .groupby('length')
        .apply(lambda x: x.sample(min(len(x), 5), random_state=42))
        .reset_index(drop=True)
    )

    # Visualization
    unique_lengths = sorted(sampled_paths['length'].unique())

    plt.figure(figsize=(15, 45))

    plot_idx = 1
    for length in unique_lengths:
        # Extract paths of this length
        length_group = sampled_paths[sampled_paths['length'] == length]['distances']

        # Plot sampled paths with individual highlighting
        for i, distances in enumerate(length_group):
            plt.subplot(len(unique_lengths) * 10, 1, plot_idx)
            plt.title(f"Path Length {length}, Highlight {i + 1}")

            # Plot all paths in gray
            for d in length_group:
                plt.plot(d, color='gray', alpha=0.5)

            # Highlight the current path in black
            plt.plot(distances, color='black', linewidth=2)

            plt.xlim(0, length - 1)
            plt.ylabel('Semantic Distance')

            plot_idx += 1

    plt.xlabel('Path Steps')
    plt.tight_layout()
    plt.show()

    return distance_df, path_df, sampled_paths


