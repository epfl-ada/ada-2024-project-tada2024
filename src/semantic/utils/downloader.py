import os
import urllib.request
import tarfile
import argparse


def download_dataset(dataset_dir):
    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # URLs to download
    urls = [
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz",
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_html.tar.gz",
    ]

    # Download and extract each file
    for url in urls:
        file_name = url.split("/")[-1]
        file_path = os.path.join(dataset_dir, file_name)

        # Download the file
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(url, file_path)

        # Extract the file
        print(f"Extracting {file_name}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=dataset_dir)

        # Remove the tar.gz file
        print(f"Removing {file_name}...")
        os.remove(file_path)

    print("Download and extraction completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADA text embedding")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/wikispeedia",
        help="Directory to store the dataset",
    )
    args = parser.parse_args()
    download_dataset(args.dataset_dir)
