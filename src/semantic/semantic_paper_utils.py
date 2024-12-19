import math
from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def process_wikispeedia_data(articles_file_path, paths_file_path, links_file_path):
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

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(avg_inf_gain)
    plt.grid(True)
    plt.show()

    return 
    
