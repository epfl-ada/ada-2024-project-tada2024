import pandas as pd
import matplotlib.pyplot as plt

def plot_scaled_scores():
    # Load dataframes
    bird_to_hitler_semantic_mpnet = pd.read_csv('data/semantic/semantic_mpnet.csv').iloc[[2136, 2140, 2173, 2212]]
    bird_to_hitler_semantic_paper = pd.read_csv('data/semantic/semantic_paper.csv').iloc[[2136, 2140, 2173, 2212]]
    bird_to_hitler_emotion_surprise = pd.read_csv('data/emotion/backtracking_path_weighted_emotions_surprise.csv').iloc[[2136, 2140, 2173, 2212]]
    bird_to_hitler_emotion_curiosity = pd.read_csv('data/emotion/backtracking_path_weighted_emotions_curiosity.csv').iloc[[2136, 2140, 2173, 2212]]

    # Format and rename columns
    bird_to_hitler_semantic_mpnet = bird_to_hitler_semantic_mpnet.loc[:, 'distance_1':'distance_4']
    bird_to_hitler_semantic_mpnet.rename(columns={
        'distance_1': 'Step_1',
        'distance_2': 'Step_2',
        'distance_3': 'Step_3',
        'distance_4': 'Step_4'
    }, inplace=True)

    bird_to_hitler_semantic_paper = bird_to_hitler_semantic_paper.loc[:, 'Step_1':'Step_4']
    bird_to_hitler_emotion_surprise = bird_to_hitler_emotion_surprise.loc[:, 'Step_1':'Step_4']
    bird_to_hitler_emotion_curiosity = bird_to_hitler_emotion_curiosity.loc[:, 'Step_1':'Step_4']

    # Store dataframes in a dictionary
    dataframes = {
        "bird_to_hitler_emotion_curiosity": bird_to_hitler_emotion_curiosity,
        "bird_to_hitler_emotion_surprise": bird_to_hitler_emotion_surprise,
        "bird_to_hitler_semantic_mpnet": bird_to_hitler_semantic_mpnet,
        "bird_to_hitler_semantic_paper": bird_to_hitler_semantic_paper,
    }

    # Plot each dataframe
    for df_name, df in dataframes.items():
        min_val = df.min().min()
        max_val = df.max().max()

        # Scale the dataframe
        scaled_df = (df - min_val) / (max_val - min_val)

        # Plot each row
        plt.figure()
        for index, row in scaled_df.iterrows():
            plt.plot(scaled_df.columns, row, marker='o', label=f"Row {index}")
            for x, y in zip(scaled_df.columns, row):
                plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')

        plt.xlabel("Steps")
        plt.ylabel("Scaled Scores")
        plt.title(f"Scaled Scores for {df_name.replace('_', ' ').capitalize()}")
        plt.legend()
        plt.show()

    # Plot comparison of common paths across all dataframes
    all_paths = set.intersection(*(set(df.index) for df in dataframes.values()))

    for path in all_paths:
        plt.figure(figsize=(10, 6))

        for df_name, df in dataframes.items():
            if path in df.index:
                row = df.loc[path]

                # Normalize the row
                min_val = row.min()
                max_val = row.max()
                scaled_row = (row - min_val) / (max_val - min_val)

                # Plot the row
                plt.plot(
                    scaled_row.index,
                    scaled_row.values,
                    marker='o',
                    label=f"{df_name.replace('_', ' ').capitalize()}"
                )
                for x, y in zip(scaled_row.index, scaled_row.values):
                    plt.text(x, y, f"{y:.2f}", fontsize=8, ha='center', va='bottom')

        plt.xlabel("Steps")
        plt.ylabel("Scaled Scores")
        plt.title(f"Comparison of '{path}' Across Dataframes")
        plt.legend()
        plt.show()


