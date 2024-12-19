import os
import re
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from scipy.stats import pearsonr


# The following codes are used to preprocess the text data extracted from the HTML files.

## Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

## File paths
html_dir = 'data/original_dataset/wpcd/wp'
paths_dir = 'data/original_dataset/wikispeedia_paths-and-graph/paths_finished.tsv'

## Functions
def preprocess_text(text):
    """
    Tokenizes and lemmatizes each word in the text.
    """
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def extract_text_from_html(html_content):
    """
    Extracts text from HTML content containing hyperlinks with anchor text,
    and preprocess_text(text) and splits it into sentences.
    """

    soup = BeautifulSoup(html_content, 'html.parser')

    # Process anchor tags to keep hyperlink text and URL
    for a in soup.find_all('a', href=True):

        # Replace each anchor tag with '[anchor_text](url)' format
        a.replace_with(f"[{a.get_text()}](URL)")

    # Get the full text with hyperlinks
    text = ' '.join(soup.stripped_strings) 

    # Split text into sentences based on punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Apply preprocessing (lemmatization) to each sentence
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]   
    return processed_sentences

def extract_sentences(html_dir, if_save=True):
    """
    Process HTML files in a directory to extract text 
    and save the results to a CSV file.
    """

    # If if_save is False, run the code, by default it will not run.
    if not if_save:

        # Store extracted text in df
        df = pd.DataFrame()

        # Define the directory containing HTML files
        extraction_path = html_dir   

        # Calculate the total number of HTML files for tracking the progress
        total_files = sum(
            len([name for name in os.listdir(os.path.join(extraction_path, folder)) if name.endswith(('.html', '.htm'))])
            for folder in os.listdir(extraction_path)
            if os.path.isdir(os.path.join(extraction_path, folder))
        )

        # The number of processed files
        processed_files = 0
        
        # Iterate through each folder in the directory
        for folder_name in os.listdir(extraction_path):
            folder_path = os.path.join(extraction_path, folder_name)
            
            # Check if the folder path is a directory
            if os.path.isdir(folder_path):
                
                # Iterate through each file in the folder
                for file_name in os.listdir(folder_path):
                    
                    # Process only files with .html or .htm extensions
                    if file_name.endswith(('.html', '.htm')):
                        file_path = os.path.join(folder_path, file_name)
                        
                        # Read file content, handling potential encoding issues
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                html_content = file.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as file:
                                html_content = file.read()
                        
                        # Extract sentences using extract_text_from_html function
                        sentences = extract_text_from_html(html_content)
                        
                        # Create a temporary DataFrame for the extracted sentences
                        temp_df = pd.DataFrame([sentences], index=[file_name])
                        
                        # Append the temporary DataFrame to the main DataFrame
                        df = pd.concat([df, temp_df])
                        
                        # the processed files counter increased by 1 
                        processed_files += 1
                        
                        # Print progress every 500 files
                        if processed_files % 500 == 0:
                            print(f"Processed {processed_files}/{total_files} files")
        
        # Save the extracted sentences to a CSV file
        df.to_csv('data/emotion/extracted_sentences.csv', index=True)
        print('Preprocessing finished and sentence corpus saved.')
    
    # read the existing CSV file with extracted sentences
    else:
        df = pd.read_csv('data/emotion/extracted_sentences.csv')
    
    # Return the DataFrame containing extracted sentences
    return df


# The following code is to generate a frequency dictionary of links between articles on user paths.
def generate_link_freq_dict(paths_dir, if_save=True):
    """
    Generate a frequency dictionary of links between articles based on user paths
    and save the result as a JSON file.
    """

    # If if_save is False, run the code, by default it will not run.
    if not if_save:
        # Load data
        file_path = paths_dir
        data = pd.read_csv(
            file_path,
            sep='\t',              
            comment='#',           
            header=0,              
            names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]  # Specify column names
        )
        
        # Store link counts
        target_count_dict = defaultdict(lambda: defaultdict(int))
        
        # Iterate through the 'path' column in the dataset
        for path in data['path']:
            # Split the semicolon-separated path into individual articles
            articles = path.split(';')
            
            # Only process paths that contain articles
            if articles:
                # The last article in the path is the target article
                target_article = articles[-1]
                
                # Iterate over all other articles in the path, excluding the target
                for article in articles[:-1]:
                    # Increment the count for the link between the current and the target article
                    target_count_dict[target_article][article] += 1
        
        # Convert the nested defaultdict into a standard dictionary for saving
        target_count_dict = {target: dict(counts) for target, counts in target_count_dict.items()}
        
        # Define the output path for the JSON file
        output_path = 'data/emotion/link_freq_dict.json'
        
        # Save the frequency dictionary as a JSON file
        with open(output_path, 'w') as f:
            json.dump(target_count_dict, f)
        
        print("Dictionary saved as link_freq_dict.json")
    
    # Example output structure of the frequency dictionary
    print("{'African_slave_trade': {'14th_century': 2, 'Europe': 3, 'Africa': 16, 'Atlantic_slave_trade': 18, ......}")


# The following code is used to extract paths from the dataset and save them to a CSV file.
def extract_path_data(input_file_path, handle_backtracking='remove', if_save=True):
    """
    Extracts paths from the dataset and saves them to a CSV file. you can specify how to handle backtracking.
    """
    # Determine the output file path based on the backtracking mode
    if handle_backtracking == 'remove':
        output_file_path = 'data/emotion/extracted_paths_no_backtracking.csv'
    elif handle_backtracking == 'replace':
        output_file_path = 'data/emotion/extracted_paths_with_backtracking.csv'
    else:
        raise ValueError("Invalid value for handle_backtracking. Use 'remove' or 'replace'.")
    if not if_save:

        # Load the CSV file containing the paths
        data = pd.read_csv(input_file_path, sep='\t', header=None, names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])
        data = data.dropna(subset=['path'])

        path_data = []
        for path in data['path']:
            articles = path.split(';')
            cleaned_path = []

            for article in articles:
                if article == '<':
                    if handle_backtracking == 'replace' and cleaned_path:
                        # Replace '<' with the previous article in the path
                        cleaned_path.append(cleaned_path[-1])
                    # If 'remove', do nothing (simply skip the '<' entry)
                else:
                    # Add non-backtracking article to path
                    cleaned_path.append(article)

            target = cleaned_path[-1] if cleaned_path else None  # Last article as target
            path_indices = cleaned_path[:-1]  # All but the last are indexed articles

            # Structure row data with target first, then indexed articles
            row_data = {'target': target}
            row_data.update({f'Index_{i+1}': article for i, article in enumerate(path_indices)})
            path_data.append(row_data)

        # Convert list of dictionaries to a DataFrame
        path_df = pd.DataFrame(path_data)
        path_df.to_csv(output_file_path, index=False)

    print(f"Paths have been extracted and saved to {output_file_path}")


# Calculate relative emotion jump rates for a given emotion and save to a CSV.
def calculate_emotion_jump_rates(emotion_df, emotion_name, output_path, alpha1=0.1, alpha2=0.5):
    """
    Calculate relative emotion jump rates for a given emotion and save to a CSV.
    """
    jump_data = []
    for path_index in emotion_df.index:
        # Extract emotion values for this path, excluding the 'target' column
        emotion_values = emotion_df.loc[path_index].drop(labels=['target']).dropna()

        # Ensure no NaN or None values exist before calculation
        emotion_values = emotion_values.astype(float).dropna()

        # Compute relative differences between consecutive steps
        diffs = emotion_values.diff().abs() / emotion_values.shift(1).abs()
        diffs = diffs.dropna()  # Drop NaN values resulting from diff or division

        # Count the number of jumps in each category
        small_jumps = (diffs < alpha1).sum()
        medium_jumps = ((diffs >= alpha1) & (diffs < alpha2)).sum()
        large_jumps = (diffs >= alpha2).sum()

        total_jumps = len(diffs)

        # Append results for this path
        jump_data.append({
            'path_index': path_index,
            'small_jumps': small_jumps,
            'medium_jumps': medium_jumps,
            'large_jumps': large_jumps,
            'total_jumps': total_jumps
        })

    jump_df = pd.DataFrame(jump_data)

    # Save the jump data to CSV
    jump_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(jump_df.head())

    return jump_df