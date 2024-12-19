import os
import re
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bs4 import BeautifulSoup
from scipy.stats import pearsonr



# Calculate the weight of a sentence given a target
def calculate_sentence_weight(sentence, target, click_frequencies, W_baseline=0.2, k=0.05):
    with open(click_frequencies, 'r') as f:
        click_frequencies = json.load(f)
    
    links = re.findall(r'\[\s*.*?\s*\]\s*\(\s*URL\s*\)', sentence)
    if not links:
        return W_baseline
    target_clicks = click_frequencies.get(target, {})
    total_links_for_target = sum(target_clicks.values()) if target_clicks else 1
    link_weight_sum = sum((target_clicks.get(link, 0) / total_links_for_target) for link in links)
    weight = W_baseline + k * link_weight_sum
    return weight

# calculate_sentence_weight(sentence, target, W_baseline=0.2, k=0.05)


# Function to calculate the nested macro average for a list of weighted emotions
def nested_macro_average(weighted_emotions):
    # Initialize group sizes: [1, 2, 3, 4, ...]
    group_sizes = [1, 2, 3, 4]
    grouped_values = []
    start_idx = 0
    
    # Loop through progressively expanding groups
    while start_idx < len(weighted_emotions):
        # Determine the current group size
        group_size = group_sizes[min(len(grouped_values), len(group_sizes) - 1)]
        
        # Calculate the end index of the current group
        end_idx = min(start_idx + group_size, len(weighted_emotions))
        
        # Calculate the average weighted emotion for the current group
        group_avg = np.mean(weighted_emotions[start_idx:end_idx])
        
        # Append the average to the grouped values list
        grouped_values.append(group_avg)
        
        # Move the start index to the next group
        start_idx = end_idx
        
        # If we've exhausted the initial group sizes, continue expanding by one
        if len(grouped_values) >= len(group_sizes):
            group_sizes.append(group_sizes[-1] + 1)

    # Calculate the final Nested Macro Average across all grouped values
    nested_macro_avg = np.mean(grouped_values)
    return nested_macro_avg

# Function to calculate weighted emotions for each article in a single path, considering a specific emotion type
def calculate_weighted_emotion_for_single_path(path, target, unweighted_emotion_df, sentences_df, click_frequencies, emotion_type='surprise', W_baseline=0.2, k=0.05):
    weighted_emotions_for_steps = {'target': target}
    unweighted_emotion_df = unweighted_emotion_df.set_index(unweighted_emotion_df.columns[0])
    # sentences_df = sentences_df.set_index(sentences_df.columns[0])
    # print(sentences_df.index)
    # Loop through each article in the path
    for step_index, article in enumerate(path):
        if pd.isna(article):
            continue  # Skip if there's no article at this step
            
        # Retrieve the sentences and unweighted emotions for the current article
        sentences = sentences_df.loc[article].dropna().tolist()  # Drop any NaN values
        unweighted_emotions = unweighted_emotion_df.loc[article].dropna().tolist()  # Drop any NaN values
        
        # Calculate weighted emotions for each sentence
        weighted_emotion_list = []
        for sentence, emotion_data in zip(sentences, unweighted_emotions):
            # Find the specific emotion score based on emotion_type
            # print(list(emotion_data)
            emotion_score = next((item['score'] for item in ast.literal_eval(emotion_data)[0] if item['label'] == emotion_type), 0)
            
            # Calculate the weighted emotion
            weight = calculate_sentence_weight(sentence, target, click_frequencies, W_baseline=0.2, k=0.05)
            weighted_emotion = weight * emotion_score
            weighted_emotion_list.append(weighted_emotion)
        
        # Calculate the Nested Macro Average for the current step
        emotion_value_for_step = nested_macro_average(weighted_emotion_list)
        
        # Store the single emotion value for this step in the path
        weighted_emotions_for_steps[f'Step_{step_index+1}'] = emotion_value_for_step

    # Convert the result into a DataFrame for easy readability and manipulation
    weighted_path_df = pd.DataFrame([weighted_emotions_for_steps])
    return weighted_path_df



def process_backtracking_paths(paths_backtracking, unweighted_emotion, sentences, click_frequencies, emotion_type='surprise', W_baseline=0.2, k=0.05, if_save=True):
    if not if_save:
        # Create an empty DataFrame to store all paths' weighted emotions
        backtracking_path_weighted_emotions_df = pd.DataFrame()
        
        # Iterate over each row in paths_backtracking
        for index, row in paths_backtracking.iterrows():
            target = row['target']
            path = row.drop(labels=['target']).dropna().tolist()
            
            # Calculate the weighted emotions for this single path
            weighted_path_df = calculate_weighted_emotion_for_single_path(path, target, unweighted_emotion, sentences, click_frequencies, emotion_type, W_baseline, k)
            
            # Add an identifier column to track which path the data belongs to
            weighted_path_df['path_index'] = index
            
            # Concatenate the current path's DataFrame to the master DataFrame
            backtracking_path_weighted_emotions_df = pd.concat([backtracking_path_weighted_emotions_df, weighted_path_df], ignore_index=True)
        
        # Define the output file name based on the emotion type
        output_file = f'dataset/backtracking_path_weighted_emotions_{emotion_type}.csv'
        
        # Save the complete result to a single CSV file
        backtracking_path_weighted_emotions_df.to_csv(output_file, index=False)
    else:
        # Define the output file name based on the emotion type
        output_file = f'dataset/backtracking_path_weighted_emotions_{emotion_type}.csv' 
        print(f"Weighted emotions for each step in all paths have been calculated and saved to '{output_file}'")


