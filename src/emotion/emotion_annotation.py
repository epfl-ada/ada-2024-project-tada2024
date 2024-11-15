import pandas as pd
import re
import torch
from transformers import AutoTokenizer, pipeline

def process_csv(input_file):
    """
    Processes an input CSV file by performing the following operations:
    
    1. Cleans the input data by removing certain unwanted parts from the text.
    2. Truncates the text in each cell to a maximum of 512 tokens using a pre-trained tokenizer.
    3. Performs emotion analysis on the text using a pre-trained emotion classification model.
    4. Saves two output CSV files: one cleaned version and one processed with emotion analysis.
    
    Parameters:
    input_file (str): The path to the input CSV file.
    
    Output:
    - cleaned_extracted_sentences.csv: A CSV file with cleaned data.
    - unweighted_emotion_article.csv: A CSV file with the results of the emotion analysis.
    
    Details:
    - The function first removes anything after a '.' in the index column.
    - It removes the '( URL )' text and preceding '[...]' from other columns.
    - Then, the function tokenizes and truncates the text in all columns to a maximum of 512 tokens.
    - The emotion analysis is applied to each cell using a RoBERTa-based model.
    - Finally, the cleaned and processed CSV files are saved.
    """
    output_cleaned_file = 'cleaned_extracted_sentences.csv'
    output_processed_file = 'unweighted_emotion_article.csv'

    # Load the CSV file
    df = pd.read_csv(input_file, low_memory=False, index_col=0)

    # Remove anything after '.' and '.' itself in the first column (assuming it's the index column)
    df.index = df.index.map(lambda x: str(x).split('.')[0] if isinstance(x, str) else x)

    # Remove '( URL )' and preceding '[...]' in all other columns (excluding the first column)
    def remove_url(text):
        if isinstance(text, str):
            return re.sub(r'\[|\]\s*\(\s*URL\s*\)', '', text)
        return text

    for column in df.columns:
        df[column] = df[column].apply(remove_url)

    # Save the cleaned CSV file
    df.to_csv(output_cleaned_file)
    print(f"CSV processing complete. The cleaned file is saved as '{output_cleaned_file}'.")

    # Determine if GPU is available
    if torch.cuda.is_available():
        print("GPU is available, using CUDA...")
        device = 0
    else:
        print("GPU not available, using CPU...")
        device = -1  # CPU

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

    # Function to truncate text to 512 tokens using the tokenizer
    def truncate_text_with_tokenizer(text, tokenizer, max_length=512):
        encoded = tokenizer.encode(text, truncation=True, max_length=max_length)
        truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
        return truncated_text

    # Apply truncation with the tokenizer to ensure the tokens of content is lower than the maximum
    for column in df.columns:
        df[column] = df[column].apply(lambda x: truncate_text_with_tokenizer(x, tokenizer, max_length=512) if isinstance(x, str) else x)

    # Load model
    emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)

    # Apply the model to analyze emotions
    for column in df.columns:
        df[column] = df[column].apply(lambda x: emotion_analyzer(x) if pd.notna(x) else x)

    # Save the processed CSV file
    df.to_csv(output_processed_file)
    print(f"Emotion analysis complete. The processed file is saved as '{output_processed_file}'.")

if __name__ == "__main__":
    input_file = input("Enter the path to the input CSV file: ")
    process_csv(input_file)
