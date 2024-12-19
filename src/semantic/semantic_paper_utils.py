import math
from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


def process_wikispeedia_data(articles_file_path, paths_file_path, links_file_path, plot=False):
    # Read articles
    with open(articles_file_path, "r") as f:
        articles = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Read finished paths
    with open(paths_file_path, "r") as f:
        lines = f.read().strip().split("\n")

    processed_paths = []
    for line in lines:
        if not line or line.strip().startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) >= 4:
            items = columns[3].split(";")
            processed_path = []
            for item in items:
                if item == "<":
                    processed_path.pop()
                else:
                    processed_path.append(item)
            processed_paths.append(processed_path)

    print(f"{len(processed_paths)} paths processed.")

    # Calculate N(A = a, G = g)
    N_precalc = defaultdict(int)
    for path in processed_paths:
        goal = path[-1]
        for i in range(len(path) - 1):
            N_precalc[(path[i], goal)] += 1

    # Calculate N(A' = a', A = a, G = g)
    N_prime_precalc = defaultdict(int)
    for path in processed_paths:
        goal = path[-1]
        for i in range(len(path) - 1):
            N_prime_precalc[(path[i + 1], path[i], goal)] += 1

    # Calculate outlinks L_a
    with open(links_file_path, "r") as f:
        lines = f.read().strip().split("\n")

    outlinks = {article: set() for article in articles}
    for line in lines:
        if not line or line.strip().startswith("#"):
            continue
        columns = line.split("\t")
        outlinks[columns[0]].add(columns[1])

    L_a = {key: len(value) for key, value in outlinks.items()}

    # Calculate PageRank
    graph = nx.Graph()
    for node, neighbors in outlinks.items():
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)

    pageranks = nx.pagerank(graph)

    alpha = 1 / 5

    # Define P_star function
    def P_star(A_prime, A, G):
        return (N_prime_precalc.get((A_prime, A, G), 0) + alpha) / (
            N_precalc.get((A, G), 0) + alpha * L_a[A]
        )

    # Calculate dp
    dp = defaultdict(list)
    for path in processed_paths:
        goal = path[-1]
        for i in range(len(path) - 1):
            dp[(path[i], goal)].append(
                sum(
                    [
                        math.log2(P_star(path[j + 1], path[j], goal))
                        for j in range(i, len(path) - 1)
                    ]
                )
                / math.log2(pageranks[goal])
            )

    dist = {key: sum(value) / len(value) for key, value in dp.items()}

    # Calculate information gain
    inf_gain_list = [[] for _ in range(7)]

    for path in processed_paths[:1600]:
        for i in range(len(path) - 1):
            A = path[i]
            A_prime = path[i + 1]
            G = path[-1]

            p_star = P_star(A_prime, A, G)
            H0 = -math.log2(1 / L_a[A]) if L_a[A] > 0 else 0
            H_star = -sum(
                [
                    P_star(A_next, A, G) * math.log2(P_star(A_next, A, G))
                    for A_next in outlinks[A]
                ]
            )

            information_gain = H0 - H_star
            normalized_distance_to_goal = i / (len(path) - 1)
            index = math.floor(normalized_distance_to_goal * 7)
            inf_gain_list[index].append(information_gain)

    avg_inf_gain = [sum(l) / len(l) for l in inf_gain_list]

    if plot:
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(avg_inf_gain)
        plt.grid(True)
        plt.show()

    cut = 3/7

    dp_cut = defaultdict(list)
    for path in processed_paths:
        goal = path[-1]
        cut_index = int(len(path) * cut)
        for i in range(cut_index, len(path) - 1):
            dp_cut[(path[i], goal)].append(
                sum(
                    [
                        math.log2(P_star(path[j + 1], path[j], goal))
                        for j in range(i, len(path) - 1)
                    ]
                )
                / math.log2(pageranks[goal])
            )

    d = {key: sum(value) / len(value) for key, value in dp_cut.items()}

    
    # save distances along paths



    interpolated_distances = []
    mean_corrected_distances = []
    max_length = 0 
    for path in processed_paths:
        max_length = max(max_length, len(path))

    for path in processed_paths:
        distances = [d.get((path[i-1], path[i]), None) for i in range(1, len(path))]
        mean_corrected_distances.append(distances)
        distances += [None] * (max_length - len(distances))
        interpolated_distances.append(interpolate_nones(distances))

    interpolated = pd.DataFrame(interpolated_distances, columns=[f'Step_{i+1}' for i in range(max_length)])
    interpolated.to_csv('interpolated_paper_distances.csv', index=False)

    processed_data = column_mean_padding(mean_corrected_distances)
    mean_corrected_distances = []

    for path in processed_data:
        mean_corrected_distances.append(path + [None] * (max_length - len(path)))
        
    mean_corrected = pd.DataFrame(processed_data, columns=[f'Step_{i+1}' for i in range(max_length)])
    mean_corrected.to_csv('mean_corrected_paper_distances.csv', index=False)

    return 

def interpolate_nones(distances):
    result = distances.copy()
    n = len(distances)
    
    # Find the last non-None value
    last_valid_index = next((i for i in range(n-1, -1, -1) if distances[i] is not None), -1)
    
    for i in range(last_valid_index + 1):
        if result[i] is None:
            # Find the next non-None value
            next_valid = next((j for j in range(i+1, n) if distances[j] is not None), None)
            
            if next_valid is not None:
                prev_valid = next((j for j in range(i-1, -1, -1) if distances[j] is not None), None)
                
                if prev_valid is not None:
                    # Interpolate
                    prev_value = distances[prev_valid]
                    next_value = distances[next_valid]
                    step = (next_value - prev_value) / (next_valid - prev_valid)
                    result[i] = prev_value + step * (i - prev_valid)
    
    return result



def column_mean_padding(lists):

    deep_copied_lists = copy.deepcopy(lists)


    if not lists:
        return []

    # Find the maximum length among all lists
    max_length = max(len(lst) for lst in lists)

    # Extend shorter lists with None values
    for lst in deep_copied_lists:
        lst.extend([None] * (max_length - len(lst)))

    # Calculate column means (excluding None values)
    column_means = []
    for col in range(max_length):
        values = [lst[col] for lst in lists if col < len(lst) and lst[col] is not None]
        mean = sum(values) / len(values) if values else None
        column_means.append(mean)

    # Calculate global mean
    all_values = [val for lst in lists for val in lst if val is not None]
    global_mean = sum(all_values) / len(all_values) if all_values else 0

    # Replace None values with appropriate means
    for lst in deep_copied_lists:
        for i in range(len(lst)):
            if lst[i] is None:
                lst[i] = column_means[i] if column_means[i] is not None else global_mean

    return deep_copied_lists
