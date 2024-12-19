from src.semantic.utils.semantic_paper_util import process_wikispeedia_data


articles_path = "data/wikispeedia/wikispeedia_paths-and-graph/articles.tsv"
paths_path = "data/wikispeedia/wikispeedia_paths-and-graph/paths_finished.tsv"
links_path = "data/wikispeedia/wikispeedia_paths-and-graph/links.tsv"

process_wikispeedia_data(articles_path, paths_path, links_path)

