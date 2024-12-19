import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Function to plot the distribution of the number of links leading to target articles
def plot_link_frequencies(json_file_path):
    """
    plot the distribution of the number of links leading to target articles
    """
    # Load the dictionary from the JSON file
    with open(json_file_path, 'r') as f:
        target_count_dict = json.load(f)

    # Select a few target articles to visualize
    sample_targets = list(target_count_dict.keys())[:4]

    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Link Frequencies Leading to Target Articles (4 examples)", fontsize=16, fontweight='bold', color='midnightblue')

    # Color map for bar colors
    colors = cm.Blues(0.6)

    # Plot each target in a subplot
    for i, target in enumerate(sample_targets):
        # Get articles and frequencies, sorted by frequency
        articles = list(target_count_dict[target].keys())
        frequencies = list(target_count_dict[target].values())
        sorted_data = sorted(zip(frequencies, articles), reverse=True)
        sorted_frequencies, sorted_articles = zip(*sorted_data)

        # Choose the subplot
        ax = axs[i // 2, i % 2]
        bars = ax.bar(sorted_articles, sorted_frequencies, color=colors)

        # Styling
        ax.set_title(f"Target: {target}", fontsize=14, fontweight='semibold', color='darkblue')
        ax.set_ylabel("Frequency", fontsize=12, fontweight='semibold')
        ax.set_xticks([])  # Hide x-axis labels for clarity
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.tick_params(axis='y', colors='grey')

    # Adjust layout for a clean look
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the main title
    plt.show()

# Analyzes path lengths and showcases specific steps and paths.
def show_paths_lengths(df, start_step=1, end_step=20):
    """
    Analyzes path lengths and showcases specific steps and paths.
    """
    # Analyze path lengths
    print("Path lengths:")
    for i in range(start_step, end_step):
        current_step = f"Step_{i}"
        next_step = f"Step_{i+1}"
        count = ((df[current_step].notna()) & (df[next_step].isna())).sum()
        print(f"Length of paths {i+1}: {count}")

# Plots paths for each length on separate plots and adds a black line for the average of random samples.
def plot_paths_with_random_samples(paths_df, emotion_label, lengths=range(4, 7)):
    """
    Plots paths for each length on separate plots and adds a black line for the average of random samples.
    """
    for length in lengths:
        current_step = f"Step_{length - 1}"

        next_step = f"Step_{length}"
        valid_df = paths_df[
            paths_df[current_step].notna() &
            paths_df[next_step].isna()
        ]


        # Randomly sample 10 rows (or all rows if fewer than 10)
        sampled_df = valid_df.sample(min(10, len(valid_df)), random_state=42) # Set random_state for same as semantic

        # Create a new plot for the current size
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual sampled paths
        for _, row in sampled_df.iterrows():
            steps = row.drop(labels=['target']).dropna().astype(float)
            x = range(1, len(steps) + 1)
            ax.plot(x, steps, color='gray', alpha=0.5)

        # Compute and plot the average path for the sample
        avg_path = sampled_df.drop(columns=['target']).mean(axis=0, skipna=True)
        avg_x = range(1, len(avg_path.dropna()) + 1)
        ax.plot(avg_x, avg_path.dropna(), color='black', linewidth=2, label='Average Path')

        ax.set_title(f'Length {length} - Weighted Emotion Scores ({emotion_label})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Emotion Score')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()


# Function to calculate the correlation and p-value between two emotions for a single path
def calculate_emotion_correlation(path_index, surprise_df, curiosity_df):
    surprise_values = surprise_df.loc[path_index].drop(labels=['target']).dropna().astype(float)
    curiosity_values = curiosity_df.loc[path_index].drop(labels=['target']).dropna().astype(float)
    
    if len(surprise_values) > 1 and len(curiosity_values) > 1:
        correlation, p_value = pearsonr(surprise_values, curiosity_values)
    else:
        correlation = None  # Not enough data to calculate correlation
        p_value = None
    
    return correlation, p_value


def analyze_correlation(
    backtracking_path_weighted_emotions_surprise_df, 
    backtracking_path_weighted_emotions_curiosity_df, 
    high_corr_threshold=0.7
):
    """
    Analyze the correlation between emotions in navigation paths.
    """
    # Create a list to store correlation results
    correlation_data = []

    # Iterate over each path and calculate the correlation
    for path_index in backtracking_path_weighted_emotions_surprise_df.index:
        correlation, p_value = calculate_emotion_correlation(
            path_index,
            backtracking_path_weighted_emotions_surprise_df, 
            backtracking_path_weighted_emotions_curiosity_df
        )

        num_steps = backtracking_path_weighted_emotions_surprise_df.loc[path_index].drop(labels=['target']).dropna().shape[0]

        if correlation is not None:  # Only add rows with valid correlation values
            correlation_data.append({
                'path_index': path_index, 
                'correlation': correlation, 
                'p_value': p_value, 
                'num_steps': num_steps
            })

    # Convert the list of dictionaries to a DataFrame
    correlation_df = pd.DataFrame(correlation_data)

    # Filter for highly correlated routes
    highly_correlated_routes = correlation_df[correlation_df['correlation'].abs() > high_corr_threshold]

    # Calculate the number of highly correlated routes
    num_highly_correlated_routes = highly_correlated_routes.shape[0]

    # Calculate the proportion of entries contributing to highly correlated routes
    surprise_entries_total = backtracking_path_weighted_emotions_surprise_df.drop(columns=['target']).notna().sum().sum()
    curiosity_entries_total = backtracking_path_weighted_emotions_curiosity_df.drop(columns=['target']).notna().sum().sum()

    surprise_high_corr_entries = highly_correlated_routes['path_index'].apply(
        lambda idx: backtracking_path_weighted_emotions_surprise_df.loc[idx].drop(labels=['target']).notna().sum()
    ).sum()

    curiosity_high_corr_entries = highly_correlated_routes['path_index'].apply(
        lambda idx: backtracking_path_weighted_emotions_curiosity_df.loc[idx].drop(labels=['target']).notna().sum()
    ).sum()

    surprise_high_corr_portion = surprise_high_corr_entries / surprise_entries_total
    curiosity_high_corr_portion = curiosity_high_corr_entries / curiosity_entries_total

    # Rank by the count of num_steps in highly correlated routes
    num_steps_frequency = (
        highly_correlated_routes['num_steps']
        .value_counts()  # Count occurrences of each num_steps
        .reset_index()  # Reset index to convert to a DataFrame
        .rename(columns={'index': 'num_steps', 0: 'count'})  # Rename columns for clarity
    )

    # Ensure 'count' is numeric
    num_steps_frequency['count'] = pd.to_numeric(num_steps_frequency['count'], errors='coerce')

    # Get the top 5 num_steps values with the highest count
    top_5_high_steps = num_steps_frequency.nlargest(5, 'count')

    # Print the results for the top 5 num_steps by count
    print("Top 5 num_steps values with the highest count of highly correlated routes:")
    print(top_5_high_steps)

    # Return the results in a dictionary
    return {
        'num_highly_correlated_routes': num_highly_correlated_routes,
        'surprise_high_corr_portion': surprise_high_corr_portion,
        'curiosity_high_corr_portion': curiosity_high_corr_portion,
        'top_5_high_steps': top_5_high_steps
    }

