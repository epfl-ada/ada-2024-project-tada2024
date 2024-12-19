import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


def plot_clustering_distribution(mpnet_values, mpnet_clustering, mpnet_mapping, d1, d2):
    """
    Generate a scatter plot for clustering distribution based on two embedding dimensions.

    """
    # Extract the specified dimensions
    mpnet_values_array = np.array(mpnet_values)[:, [d1, d2]]

    # Create a DataFrame for the selected dimensions and clustering
    mpnet_df = pd.DataFrame(mpnet_values_array, columns=["d1", "d2"])
    mpnet_df['clustering'] = mpnet_clustering['K-Medoids_Cosine']['clustering']

    # Add descriptive labels to clustering
    mpnet_df['clustering'] = mpnet_df['clustering'].apply(lambda x: f"{x}-{mpnet_mapping['K-Medoids_Cosine'][x]}")

    # Plot the scatter plot
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

    # Add titles and axis labels
    plt.title('Clustering Distribution Based on d1 and d2', fontsize=14)
    plt.xlabel('d1', fontsize=12)
    plt.ylabel('d2', fontsize=12)

    # Adjust the legend position for better visibility
    plt.legend(title='Clustering', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Optimize layout to prevent overlap
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
    filtered_lengths = [3, 4, 5, 6, 7, 8, 9]
    sampled_paths = (
        path_data[path_data['length'].isin(filtered_lengths)]
        .groupby('length')
        .apply(lambda x: x.sample(min(len(x), 10), random_state=42))
        .reset_index(drop=True)
    )

    # Visualization
    unique_lengths = sorted(sampled_paths['length'].unique())

    plt.figure(figsize=(12, 20))

    for idx, length in enumerate(unique_lengths):
        plt.subplot(len(unique_lengths), 1, idx + 1)
        plt.title(f"Series {idx + 1} (Path Length {length})")

        # Extract paths of this length
        length_group = sampled_paths[sampled_paths['length'] == length]['distances']

        # Plot sampled paths
        for distances in length_group:
            plt.plot(distances, color='gray', alpha=0.5)

        # Highlight the average path
        avg_distances = np.mean(length_group.tolist(), axis=0)
        plt.plot(avg_distances, color='black', linewidth=2)

        plt.xlim(0, length - 1)
        plt.ylabel('Semantic Distance')

    plt.xlabel('Path Steps')
    plt.tight_layout()
    plt.show()

    return distance_df, path_df, sampled_paths

def plot_distances_along_path(path_semantic_distances, CUT_LENGTH=12):
''' plots Semantic Variations between Wikipeedia Articles Along User Paths'''
    with pd.option_context('future.no_silent_downcasting', True):
        filled_semantic_distances = path_semantic_distances.fillna(-1).infer_objects(copy=False)

    max_distance = np.nanmax(filled_semantic_distances.values)
    normalised_distances = filled_semantic_distances.div(max_distance)

    normalised_distances = normalised_distances.mask(normalised_distances < 0, None)
    plt.figure(figsize=(10, 6))

    for _, row in normalised_distances.iterrows():
        steps = range(CUT_LENGTH)
        plt.plot(steps, list(row)[:CUT_LENGTH], marker='o', linestyle='-', markersize=4)


    plt.title('Semantic Variations between Articles Along Paths of Wikispeedia')
    plt.xlabel('Number of Clicked Links')
    plt.ylabel('Normalised Semantic Distance between Articles')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_distances_along_path(path_semantic_distances)
