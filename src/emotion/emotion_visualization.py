import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import numpy as np
from scipy.stats import pearsonr

def plot_link_frequencies(json_file_path):
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


# Function to plot paths for given dataframe, emotion label, and axis
def plot_emotion_paths(ax, paths_df, emotion_label):
    for index, row in paths_df.iterrows():
        target = row['target']
        steps = row.drop(labels=['target']).dropna().astype(float)

        # Calculate mean and standard deviation for error bars
        means = steps.mean()
        std_devs = steps.std()

        # Plot the path with error bars
        x = range(1, len(steps) + 1)
        y = steps
        ax.plot(x, y, label=f'Path {index + 1}: {target}')

        # Fill the area between the error margins
        ax.fill_between(x, y - std_devs, y + std_devs, alpha=0.2)

    # Add labels and title for the plot
    ax.set_xlabel('Step')
    ax.set_ylabel(f'Weighted Emotion Score ({emotion_label})')
    ax.set_title(f'Weighted Emotion Scores ({emotion_label}) for the First Five Paths (error bar represent 1 standard deviation)')
    ax.legend()


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