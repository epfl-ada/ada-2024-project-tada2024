import argparse
import os
import shutil
from glob import glob

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from downloader import download_dataset


def extract_html_to_txt(
    source_folder: str, output_folder: str, valid_article_names: list
) -> list:
    """
    Function to extract text from HTML files and save them as .txt.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    # Get list of all HTML files, excluding those in the "index" subfolder
    html_files = glob(f"{source_folder}/**/*.htm", recursive=True)
    html_files = [f for f in html_files if "index" not in f.split(os.sep)]
    failed_extracts = []
    for html_file_path in tqdm(html_files):
        article_name = os.path.splitext(os.path.basename(html_file_path))[0]
        if article_name not in valid_article_names:
            continue
        output_file_path = os.path.join(output_folder, article_name + ".txt")

        cleaned_html = ""
        with open(html_file_path, "r", encoding="utf-8", errors="ignore") as html_file:
            soup = BeautifulSoup(html_file, "html.parser")

            # Remove wikipedia header
            for header in soup.find_all(attrs={"id": "siteSub"}):
                header.decompose()

            # Remove all table elements
            for table in soup.find_all("table"):
                table.decompose()

            # Remove caption of thumbnails
            for thumb in soup.find_all(attrs={"class": "thumb"}):
                thumb.decompose()

            # Add sharps for headings corresponding to heading level
            for i in range(1, 7):
                for heading in soup.find_all(f"h{i}"):
                    heading.string = ("#" * i) + heading.text

            # Remove footer
            for footer in soup.find_all(attrs={"id": "footer"}):
                footer.decompose()
            for footer in soup.find_all(attrs={"class": "printfooter"}):
                footer.decompose()

            cleaned_html = soup.prettify()

        # Write temporary cleaned html file
        with open("TMP.html", "w", encoding="utf-8", errors="ignore") as html_file:
            html_file.write(cleaned_html)

        # Load html file using chromium driver
        driver.get(f"file:///{os.path.abspath("TMP.html")}")

        # Get plain text from chromium driver
        plain_text = ""
        try:
            plain_text = driver.find_element(By.ID, "bodyContent").text
        except:
            failed_extracts.append(article_name)

        # Save plain text
        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(plain_text)

    return failed_extracts


def get_artilces_to_parse(dataset_dir) -> list:
    """
    There are additional .htm files in the wpcd directory. We will get the list of valid article names from articles.tsv to select the correct html files later
    """
    article_list_path = os.path.join(
        dataset_dir, "wikispeedia_paths-and-graph", "articles.tsv"
    )
    article_names = []
    with open(article_list_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            stripped_line = line.strip()
            if stripped_line:
                article_names.append(stripped_line)
    return article_names


def parse_argumnets():
    parser = argparse.ArgumentParser(
        description="Artile cleaning script."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Whether to download the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/wikispeedia",
        help="Directory where the dataset is stored. If --download is set, dataset will be saved in dataset_dir. (default: %(default)s)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./data/wikispeedia/wpcd/wp",
        help="Directory for the source HTML files (default: %(default)s)",
    )
    parser.add_argument(
        "--plaintext_output_dir",
        type=str,
        default="./data/semantic/output/clean_plaintext_articles",
        help="Output directory for clean plaintext articles (default: %(default)s)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argumnets()

    if args.download:
        download_dataset(args.dataset_dir)

    valid_articles = get_artilces_to_parse(args.dataset_dir)

    failed_extracts = extract_html_to_txt(
        args.source_dir, args.plaintext_output_dir, valid_articles
    )

    print("Failed to extract the following articles: ", failed_extracts)
    os.remove("TMP.html")
