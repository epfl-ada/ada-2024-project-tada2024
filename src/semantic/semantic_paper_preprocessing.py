import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
from bs4 import BeautifulSoup
import os
import re
from chardet import detect

# Define a function to load embeddings
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Define a function to calculate PCA with explained variance threshold
def calculate_pca(data, variance_threshold=0.7):
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    pca = PCA(n_components=optimal_components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data, optimal_components

# Define a function to filter embeddings
def filter_embeddings(original_embeddings, selected_keys):
    return {k: v for k, v in original_embeddings.items() if k in selected_keys}

# Save processed embeddings to a file
def save_embeddings(file_path, embeddings):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)

# Function to read and clean HTML data
def read_html_files_with_encoding(base_folder):
    """
    Recursively read HTML files from a folder, extract the content of the first <p> tag,
    and skip any folder named 'index'.
    
    :param base_folder: The top-level folder path to scan
    :return: A dictionary containing file names as keys and the first <p> text content as values
    """
    first_paragraph_texts = {}
    for root, dirs, files in os.walk(base_folder):
        # Skip directories named 'index'
        if 'index' in dirs:
            dirs.remove('index')  # Remove 'index' folder from traversal

        for file_name in files:
            # Process only HTML files
            if file_name.lower().endswith(('.html', '.htm')):  
                file_path = os.path.join(root, file_name)  # Construct the full file path

                # Auto-detect file encoding
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    result = detect(raw_data)
                    encoding = result['encoding']

                # Read HTML file content using detected encoding
                with open(file_path, 'r', encoding=encoding) as file:
                    html_content = file.read()
                
                # Parse HTML content using BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract content from all <p> tags
                paragraph_list = soup.find_all('p')
                first_paragraph = ''
                for i in range(len(paragraph_list)):
                    first_paragraph = first_paragraph + ' ' + paragraph_list[i].get_text()
                    # Remove extra whitespace
                    first_paragraph = re.sub(r'\s{2,}', ' ', first_paragraph)
                    word_count = len(first_paragraph.split())
                    # Stop once the first paragraph has at least 20 words
                    if word_count >= 20:
                        break
                
                # Store the first paragraph content with the file name as key
                first_paragraph_texts[file_name] = first_paragraph
    
    return first_paragraph_texts

# Function to save cleaned HTML data
def save_cleaned_html_data(processed_files, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(processed_files, f)

# Main function
def main():
    # File paths
    base_folder = "D://Study//Classes-M1//ADA//Milestone1 Data Cleaned//wpcd//wp"
    processed_file_name = 'new_data_cleaning//new_text_data.pkl'
    MiniLM_file = 'data/semantic/output/embeddings/all_MiniLM_L6_v2.pkl'
    mpnet_file = 'data/semantic/output/embeddings/all_mpnet_base_v2.pkl'

    # Step 1: Read and clean HTML data
    html_files = read_html_files_with_encoding(base_folder)
    
    # Process the extracted paragraphs
    processed_files = {}
    for file_name, content in html_files.items():
        # Remove the file extension, split by the last '.' character
        base_name = file_name.rsplit('.', 1)[0]  # Keep only the main file name
        
        # Strip leading and trailing spaces from the content
        cleaned_sentence = content.strip(' ')

        # Store the processed content into a new dictionary
        processed_files[base_name] = cleaned_sentence

    # Save the processed data to a pickle file
    save_cleaned_html_data(processed_files, processed_file_name)
    print(f"Processed HTML data saved to {processed_file_name}")

    # Step 2: Load embeddings
    MiniLM_original = load_embeddings(MiniLM_file)
    mpnet_original = load_embeddings(mpnet_file)

    # Example filter (keys to keep should be defined based on your dataset)
    selected_keys = set(MiniLM_original.keys()) & set(mpnet_original.keys())

    # Filter embeddings
    MiniLM_embeddings = filter_embeddings(MiniLM_original, selected_keys)
    mpnet_embeddings = filter_embeddings(mpnet_original, selected_keys)

    # Convert to array for PCA
    MiniLM_array = np.array(list(MiniLM_embeddings.values()))
    mpnet_array = np.array(list(mpnet_embeddings.values()))

    # Perform PCA
    MiniLM_pca, MiniLM_transformed, MiniLM_optimal_components = calculate_pca(MiniLM_array)
    mpnet_pca, mpnet_transformed, mpnet_optimal_components = calculate_pca(mpnet_array)

    print(f"Optimal components for MiniLM: {MiniLM_optimal_components}")
    print(f"Optimal components for mpnet: {mpnet_optimal_components}")

    # Save processed embeddings
    save_embeddings('new_data_cleaning/new_MiniLM_embedding.pkl', MiniLM_transformed)
    save_embeddings('new_data_cleaning/new_mpnet_embedding.pkl', mpnet_transformed)

    # Save PCA models for future use
    save_embeddings('new_data_cleaning/pca_MiniLM_model.pkl', MiniLM_pca)
    save_embeddings('new_data_cleaning/pca_mpnet_model.pkl', mpnet_pca)

if __name__ == "__main__":
    main()
