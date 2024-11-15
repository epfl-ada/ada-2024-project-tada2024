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

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# File paths
html_dir = 'data/original_dataset/wpcd/wp'
paths_dir = 'data/original_dataset/wikispeedia_paths-and-graph/paths_finished.tsv'
output_sentence_csv = 'data/emotion/extracted_sentences.csv'

def preprocess_text(text):
    """
    Tokenizes and lemmatizes each word in the text.
    """
    
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def extract_text_from_html(html_content):
    """
    Extracts readable text from HTML content, preserves hyperlinks with anchor text,
    performs lemmatization and stemming, and splits it into sentences.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Process anchor tags to preserve hyperlink text and URL
    for a in soup.find_all('a', href=True):
        # Replace each anchor tag with '[anchor_text](url)' format
        a.replace_with(f"[{a.get_text()}](URL)")
    
    # Get the full text with preserved hyperlinks
    text = ' '.join(soup.stripped_strings)
    
    # Split text into sentences based on punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Apply preprocessing (lemmatization and stemming) to each sentence
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    return processed_sentences

# Process each HTML file, extract sentences

def extract_sentences(html_dir, if_save=True):
    if not if_save:
        df = pd.DataFrame()
        extraction_path = html_dir
        total_files = sum(
            len([name for name in os.listdir(os.path.join(extraction_path, folder)) if name.endswith(('.html', '.htm'))])
            for folder in os.listdir(extraction_path)
            if os.path.isdir(os.path.join(extraction_path, folder))
        )
        processed_files = 0
        # print("preprocessing starts")
        for folder_name in os.listdir(extraction_path):
            folder_path = os.path.join(extraction_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.html', '.htm')):
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                html_content = file.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as file:
                                html_content = file.read()
                        sentences = extract_text_from_html(html_content)
                        temp_df = pd.DataFrame([sentences], index=[file_name])
                        df = pd.concat([df, temp_df])
                        processed_files += 1
                        # if processed_files % 500 == 0:
                        #     print(f"Processed {processed_files}/{total_files} files")
        df.to_csv(output_sentence_csv, index=True)
        print('Preprocessing finished and sentence corpus saved.')
    else:
        df = pd.read_csv(output_sentence_csv)
    return df
    

# Generate link count frequency dictionary
def generate_link_freq_dict(paths_dir, if_save=True):
    if not if_save:
        file_path = paths_dir
        data = pd.read_csv(file_path, sep='\t', comment='#', header=0, names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"])
        target_count_dict = defaultdict(lambda: defaultdict(int))
        for path in data['path']:
            articles = path.split(';')
            if articles:
                target_article = articles[-1]
                for article in articles[:-1]:
                    target_count_dict[target_article][article] += 1
        target_count_dict = {target: dict(counts) for target, counts in target_count_dict.items()}
        output_path = 'data/emotion/link_freq_dict.json'
        with open(output_path, 'w') as f:
            json.dump(target_count_dict, f)
        print("Dictionary saved as link_freq_dict.json")
    print("{'African_slave_trade': {'14th_century': 2,'Europe': 3,'Africa': 16,'Atlantic_slave_trade': 18, ......}")



# extract_paths.py
def extract_path_data(input_file_path, handle_backtracking='remove', if_save=True):
    # Determine the output file path based on the backtracking mode
    if handle_backtracking == 'remove':
        output_file_path = 'dataset/extracted_paths_no_backtracking.csv'
    elif handle_backtracking == 'replace':
        output_file_path = 'dataset/extracted_paths_with_backtracking.csv'
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

