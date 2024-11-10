# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart
```bash
# clone project
git clone <project link>
cd <project repo>

# install requirements
pip install -r pip_requirements.txt
```

### How to use the project

#### Notebooks
Each notebook, starts with a configuration cell to define dataset directory, output directory and whether or not to download the dataset. 

##### Plaintext cleaning (clean_plaintext_articles.ipynb)
This notebook uses the HTML files of each article to extract their plaintext and save it as a .txt file. Currently, it only considers the inner text of paragraph tags. There are some known issues, and further cleaning is needed.

##### Generate Embeddings (generate_embeddings.ipynb)
In this notebook, given a predefined model, all plaintext articles are embedded and stored in a dictionary. Finally, the dictionary is saved locally as a .pkl file.

## Project Structure

The directory structure of new project looks like this:

```
├── data                            <- Project data files
│
├── src                             <- Source code
│   ├── embedding_models            <- Contains classes for embedding models
│   ├── downloader.py               <- Utility to download datasets
│
├── tests                           <- Tests of any kind
│
├── results.ipynb                   <- a well-structured notebook showing the results
├── clean_article_plaintext.ipynb   <- Plaintext cleaning notebook
├── generate_embeddings.ipynb       <- Article embedding generator
├── .gitignore                      <- List of files ignored by git
├── pip_requirements.txt            <- File for installing python dependencies
└── README.md
```

