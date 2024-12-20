import pandas as pd
import numpy as np

def calculate_rowwise_correlation(df1, df2, drop_columns1=None, drop_columns2=None):
    """
    Calculate row-wise correlations between two DataFrames after dropping specified columns.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        drop_columns1 (list, optional): Columns to drop from the first DataFrame.
        drop_columns2 (list, optional): Columns to drop from the second DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the row-wise correlations.
    """
    # Drop specified columns from each DataFrame
    numeric_df1 = df1.drop(columns=drop_columns1, errors='ignore') if drop_columns1 else df1
    numeric_df2 = df2.drop(columns=drop_columns2, errors='ignore') if drop_columns2 else df2

    # Ensure the two DataFrames have the same number of rows
    if numeric_df1.shape[0] != numeric_df2.shape[0]:
        raise ValueError("The two DataFrames must have the same number of rows.")

    # Function to calculate correlation for a single row pair
    def calculate_row_correlation(row1, row2):
        # Drop NaN values from both rows
        valid_indices = ~np.isnan(row1) & ~np.isnan(row2)
        if valid_indices.sum() > 1:  # At least two points are needed to calculate correlation
            return np.corrcoef(row1[valid_indices], row2[valid_indices])[0, 1]
        else:
            return np.nan  # Return NaN if not enough data points

    # Calculate correlation line by line
    correlations = [
        calculate_row_correlation(numeric_df1.iloc[i].values, numeric_df2.iloc[i].values)
        for i in range(numeric_df1.shape[0])
    ]

    # Convert the results to a DataFrame for better visualization and analysis
    return pd.DataFrame({"correlation": correlations})



def mean_correlation(df, name=None):
    """
    Calculate the mean correlation using Fisher's z-transformation for the 'correlation' column,
    handling values at the boundary of [-1, 1]. Includes a print statement for debugging.
    """
    # Extract the 'correlation' column
    correlations = df[name]
    
    # Clip values slightly to avoid domain issues with np.arctanh
    correlations = correlations.clip(-0.9999, 0.9999)
    
    # Apply Fisher's z-transformation
    z_transformed = np.arctanh(correlations)
    
    # Compute the mean of the z-transformed values
    z_mean = np.mean(z_transformed)
    
    # Convert back to the correlation scale
    mean_corr = np.tanh(z_mean)
    
    # Print the DataFrame name and the calculated mean correlation
    if name:
        print(f"{name} Mean correlation: {mean_corr}")
    

def process_and_merge_data(paths_file, output_file,
                           mpnet_surprise_correlation_df,
                           mpnet_curiosity_correlation_df,
                           paper_surprise_correlation_df,
                           paper_curiosity_correlation_df,
                           surprise_jump_df, curiosity_jump_df, if_save=True):
    """
    Process the paths_finished dataset, calculate path lengths, and merge with other correlation data.
    """
    # Load and process paths data
    df = pd.read_csv(paths_file, sep='\t', header=None, 
                     names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])
    df = df.dropna(subset=['path'])
    df['path_length'] = df['path'].apply(lambda x: len(x.split(';')))
    df = df[['path', 'path_length']].reset_index(drop=True)

    # Merge with correlation data
    merged_df = pd.concat(
        [df, mpnet_surprise_correlation_df, mpnet_curiosity_correlation_df, 
         paper_surprise_correlation_df, paper_curiosity_correlation_df, 
         surprise_jump_df, curiosity_jump_df],
        axis=1,
    )

    # Rename columns for clarity
    merged_df.columns = [
        'path', 'path_length', 
        'mpnet_surprise_corr', 'mpnet_curiosity_corr', 
        'paper_surprise_corr', 'paper_curiosity_corr', 
        'surprise_small_jumps_rate', 'surprise_medium_jumps_rate', 'surprise_large_jumps_rate', 
        'curiosity_small_jumps_rate', 'curiosity_medium_jumps_rate', 'curiosity_large_jumps_rate'
    ]

    # Save to CSV
    if not if_save:
        merged_df.to_csv(output_file, index=False)

    merged_df.head(5)
    # Return the head of the DataFrame for quick inspection
    return merged_df



def analyze_and_filter_correlations(merged_df, if_save=True):
    # Define a helper function to calculate percentage and filter rows
    def calculate_percentage_and_filter(df, conditions):
        filtered_df = df[np.logical_and.reduce(conditions)]
        percentage = (len(filtered_df) / len(df)) * 100
        return filtered_df, percentage

    # Define the conditions for each correlation type
    surprise_conditions = [
        np.abs(merged_df['mpnet_surprise_corr']) > 0.7,
        np.abs(merged_df['paper_surprise_corr']) > 0.7
    ]

    curiosity_conditions = [
        np.abs(merged_df['mpnet_curiosity_corr']) > 0.7,
        np.abs(merged_df['paper_curiosity_corr']) > 0.7
    ]

    combined_conditions = surprise_conditions + curiosity_conditions

    # Analyze surprise correlations
    surprise_high_corr_df, surprise_percentage = calculate_percentage_and_filter(merged_df, surprise_conditions)
    print(f"Percentage of rows where surprise conditions are met: {surprise_percentage:.2f}%")

    # Analyze curiosity correlations
    curiosity_high_corr_df, curiosity_percentage = calculate_percentage_and_filter(merged_df, curiosity_conditions)
    print(f"Percentage of rows where curiosity conditions are met: {curiosity_percentage:.2f}%")

    # Analyze combined correlations
    surprise_curiosity_high_corr_df, combined_percentage = calculate_percentage_and_filter(merged_df, combined_conditions)
    print(f"Percentage of rows where combined conditions are met: {combined_percentage:.2f}%")

    # Filter rows with path length of 5 from the combined dataframe
    selected_path_lengths_df = surprise_curiosity_high_corr_df[
        surprise_curiosity_high_corr_df['path_length'].isin([5])
    ]

    # Save the filtered DataFrame to a CSV file
    if not if_save:
      selected_path_lengths_df.to_csv('data/correlation/selected_path_lengths_df.csv', index=False)
    print(selected_path_lengths_df.head(3))

    return 