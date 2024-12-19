import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.patches import FancyArrowPatch

def plot_semantic_network(csv_path, main_node, samples, random_state=42, distance_scale_factor=2.0, figsize=(10, 10), dpi=300):
    """
    Plot a semantic network with a central node and its connections, 
    where distances determine the positioning of nodes.
    """
    # Load and preprocess data
    df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)
    reduced_df = df[df["a"] == main_node].sample(samples, random_state=random_state).sort_values("distance").copy()

    # Create graph
    G = nx.Graph()
    G.add_node(main_node)
    for _, row in reduced_df.iterrows():
        G.add_node(row['b'])
        G.add_edge(main_node, row['b'], weight=row['distance'])

    # Position nodes
    pos = {main_node: (0, 0)}
    num_words = len(reduced_df)
    angles = np.linspace(0, 2 * np.pi, num_words, endpoint=False)

    for angle, (_, row) in zip(angles, reduced_df.iterrows()):
        r = row['distance'] * distance_scale_factor
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pos[row['b']] = (x, y)

    # Plot the graph
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    node_sizes = [1000 if node == main_node else 50 for node in G.nodes()]
    node_colors = ["#FF6F61" if node == main_node else "#6FB1FF" for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, linewidths=1, edgecolors='white', ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="#888888", ax=ax)

    # Add labels
    for angle, (_, row) in zip(angles, reduced_df.iterrows()):
        distance_val = row['distance']
        label_text = f"{row['b']} ({distance_val:.2f})"
        angle_degs = np.degrees(angle)

        offset = 0.2
        r = row['distance'] * distance_scale_factor
        x_offset = (r + offset) * np.cos(angle)
        y_offset = (r + offset) * np.sin(angle)

        if 90 < angle_degs <= 270:
            rotation = angle_degs + 180
            ha = 'right'
        else:
            rotation = angle_degs
            ha = 'left'

        ax.text(x_offset, y_offset, label_text,
                rotation=rotation,
                rotation_mode='anchor',
                fontsize=8,
                color='black',
                va='center', ha=ha)

    # Center label for the main node
    ax.text(0, 0, main_node, fontsize=8, color="black", ha='center', va='center')

    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.tight_layout()
    plt.show()





def visualize_distance_graph(
    main_node="Mathematics",
    samples=20,
    random_state=16,
    distance_file="data/semantic/distances.csv",
    filtered_file="data/semantic/distances_filtered.csv"
):
    # Load data
    df = pd.read_csv(distance_file, index_col=0).reset_index(drop=True)
    df_cut = pd.read_csv(filtered_file, index_col=0).reset_index(drop=True)

    # Merge and filter data
    merge_df = df.merge(df_cut, on=["a", "b"], how="right", suffixes=(None, "_cut"))
    merge_df = merge_df[merge_df["a"] == main_node]
    merge_df = merge_df.sort_values("distance")
    merge_df = merge_df.sample(samples, random_state=random_state)

    red_list = merge_df[merge_df["distance"] != merge_df["distance_cut"]]["b"].to_list()

    show_df = merge_df.copy()

    # Filter rows where "b" is in the red_list and update
    filtered_rows = merge_df[merge_df["b"].isin(red_list)].copy()
    filtered_rows["b"] = filtered_rows["b"] + "_CUT"
    filtered_rows["distance"] = filtered_rows["distance_cut"]

    # Append to show_df
    show_df = pd.concat([show_df, filtered_rows], ignore_index=True).sort_values("distance")

    # Update green_list
    green_list = filtered_rows["b"].tolist()

    # Create the graph
    G = nx.Graph()
    G.add_node(main_node)

    for _, row in show_df.iterrows():
        G.add_node(row['b'])
        G.add_edge(main_node, row['b'], weight=row['distance'])

    # Position nodes radially
    distance_scale_factor = 2.0
    pos = {main_node: (0, 0)}

    num_words = len(show_df)
    angles = np.linspace(0, 2 * np.pi, num_words, endpoint=False)

    for angle, (_, row) in zip(angles, show_df.iterrows()):
        r = row['distance'] * distance_scale_factor
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pos[row['b']] = (x, y)

    # Split edges into categories
    red_edges = [(main_node, row['b']) for _, row in show_df.iterrows() if row['b'] in red_list]
    green_edges = [(main_node, row['b']) for _, row in show_df.iterrows() if row['b'] in green_list]
    normal_edges = [(main_node, row['b']) for _, row in show_df.iterrows()
                    if row['b'] not in red_list and row['b'] not in green_list]

    # Plot settings
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    node_sizes = [300 if node == main_node else 20 for node in G.nodes()]
    node_colors = ["#FF6F61" if node == main_node else "#6FB1FF" for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, linewidths=1, edgecolors='white', ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1.5, edge_color="#888888", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, width=1.5, edge_color="red", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, width=1.5, edge_color="lightgreen", ax=ax)

    # Draw labels
    for angle, (_, row) in zip(angles, show_df.iterrows()):
        distance_val = row['distance']
        label_text = f"{row['b']} ({distance_val:.2f})"
        angle_degs = np.degrees(angle)

        offset = 0.3
        r = row['distance'] * distance_scale_factor
        x_offset = (r + offset) * np.cos(angle)
        y_offset = (r + offset) * np.sin(angle)

        if 90 < angle_degs <= 270:
            rotation = angle_degs + 180
            ha = 'right'
        else:
            rotation = angle_degs
            ha = 'left'

        ax.text(
            x_offset, y_offset, label_text,
            rotation=rotation,
            rotation_mode='anchor',
            fontsize=8,
            color='black',
            va='center', ha=ha
        )

    # Draw curved arrows
    for red_node in red_list:
        green_node = f"{red_node}_CUT"
        if red_node in pos and green_node in pos:
            red_pos = pos[red_node]
            green_pos = pos[green_node]

            angle_red = np.degrees(np.arctan2(red_pos[1], red_pos[0])) % 360
            angle_green = np.degrees(np.arctan2(green_pos[1], green_pos[0])) % 360

            if angle_green > angle_red:
                base_radius = np.linalg.norm(red_pos)
                green_pos = (base_radius * np.cos(np.radians(angle_green)),
                             base_radius * np.sin(np.radians(angle_green)))
                arrow = FancyArrowPatch(
                    red_pos, green_pos,
                    connectionstyle=f"arc3,rad={np.power(base_radius, 1/20) / 2}",
                    arrowstyle="->", color="purple", lw=1, mutation_scale=10, linestyle=":"
                )
            else:
                base_radius = np.linalg.norm(green_pos)
                red_pos = (base_radius * np.cos(np.radians(angle_red)),
                           base_radius * np.sin(np.radians(angle_red)))
                arrow = FancyArrowPatch(
                    red_pos, green_pos,
                    connectionstyle=f"arc3,rad={-np.power(base_radius, 1/20) / 2}",
                    arrowstyle="->", color="purple", lw=1, mutation_scale=10, linestyle=":"
                )
            ax.add_patch(arrow)

    # Label the main node
    ax.text(0, 0, main_node, fontsize=8, color="black", ha='center', va='center')

    # Final plot settings
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.tight_layout()
    plt.show()




import matplotlib.pyplot as plt

def plot_distances_along_path(CUT_LENGTH=12):
    path_semantic_distances = pd.read_csv('data/semantic/semantic_paper.csv')
    with pd.option_context('future.no_silent_downcasting', True):
        filled_semantic_distances = path_semantic_distances.fillna(-1).infer_objects(copy=False)

    max_distance = np.nanmax(filled_semantic_distances.values)
    normalised_distances = filled_semantic_distances.div(max_distance)

    normalised_distances = normalised_distances.mask(normalised_distances < 0, None)
    plt.figure(figsize=(10, 6))

    for _, row in normalised_distances.iterrows():
        steps = range(CUT_LENGTH)
        plt.plot(steps, list(row)[:CUT_LENGTH], marker='o', linestyle='-', markersize=4)


    plt.title('Semantic Variations between Articles Along Paths of Wikispeedia')
    plt.xlabel('Number of Clicked Links')
    plt.ylabel('Normalised Semantic Distance between Articles')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

