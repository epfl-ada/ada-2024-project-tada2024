import argparse
import os
import pickle
from datetime import datetime
from glob import glob

from embedding_models.all_MiniLM_L6_v2 import All_MiniLM_L6_v2
from embedding_models.all_mpnet_base_v2 import all_mpnet_base_v2
from embedding_models.roberta import Roberta
from tqdm import tqdm

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_embeddings(articles_dir, model):
    embeddings = {}
    for article_path in tqdm(glob(f"{articles_dir}/*.txt")):
        article_name = os.path.splitext(os.path.basename(article_path))[0]
        with open(article_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
            embedding = model.embed(text)
            embeddings[article_name] = embedding

    return embeddings


def save_embeddings(embeddings, output_dir, model_name, overwrite_latest=False):
    # Create embedding output directory
    embeddings_output_dir = os.path.join(output_dir, model_name, current_datetime)
    if not os.path.exists(embeddings_output_dir):
        os.makedirs(embeddings_output_dir)

    # Save pkl embeddings in output dir
    embeddings_output_path = os.path.join(embeddings_output_dir, "embeddings.pkl")
    with open(embeddings_output_path, "wb") as file:
        pickle.dump(embeddings, file)

    if overwrite_latest:
        # Save pkl embeddings as the latest one
        latest_embedding_output_path = os.path.join(output_dir, "latest_embeddings.pkl")
        with open(latest_embedding_output_path, "wb") as file:
            pickle.dump(embeddings, file)


def parse_argumnets():
    parser = argparse.ArgumentParser(description="Embedding Generator script.")

    parser.add_argument(
        "--articles_dir",
        type=str,
        default="./data/semantic/output/clean_plaintext_articles",
        help="Directory where the plaintext articles are stored (default: %(default)s)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="./data/semantic/output/embeddings",
        help="Directory to store embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all_MiniLM_L6_v2",
        choices=["all_MiniLM_L6_v2", "all_mpnet_base_v2", "roberta"],
        help="Embedding Model Name (default: %(default)s)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argumnets()

    model = None
    if args.model_name == "all_MiniLM_L6_v2":
        model = All_MiniLM_L6_v2()
    elif args.model_name == "all_mpnet_base_v2":
        model = all_mpnet_base_v2()
    elif args.model_name == "roberta":
        model = Roberta()

    article_embeddings = generate_embeddings(args.articles_dir, model)

    save_embeddings(
        article_embeddings, args.embeddings_dir, args.model_name, overwrite_latest=True
    )
