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
    correlations = df['correlation']
    
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
    

