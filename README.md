# Your project name

This is a template repo for your project to help you organise and document your code better.
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# conda
conda create -n <env_name> python=3.10
conda activate <env_name>

# install requirements
pip install -r pip_requirements.txt
```

### How to use the project

#### Semantic Scripts

Each notebook, starts with a configuration cell to define dataset directory, output directory and whether or not to download the dataset.

##### Plaintext cleaning (clean_articels.py)

This script uses the HTML files of each article to extract their plaintext and save it as a .txt file.
Specify `--download` if the dataset has never been downloaded.

```bash
python ./src/semantic/clean_articles.py --download 
```

##### Generate Embeddings (generate_embeddings.py)

In this script, given a predefined model, all plaintext articles are embedded and stored in a dictionary. Finally, the dictionary is saved locally as a .pkl file.

```bash
python ./src/semantic/generate_embeddings.py --model_name "all_mpnet_base_v2" 
```



##### Perform Clustering (perform_clustering.py)

In this script, given the embedding results, the clusterings are calculated and saved locally as a .pkl file.

```bash
python ./src/semantic/perform_clustering.py --embedding_model_name "all_mpnet_base_v2" 
```


## Project Structure

The directory structure of new project looks like this:

```
.
├── README.md
├── data
│   ├── semantic
│   │   └── output
│   │       ├── clean_plaintext_articles	-> contains cleaned articles from HTML files
│   │       ├── clustering			-> contains results of clusterings for each embedding model
│   │       │   ├── all_MiniLM_L6_v2.pkl
│   │       │   ├── all_mpnet_base_v2.pkl
│   │       │   └── roberta.pkl
│   │       └── embeddings			-> contains embeddings of articles using different models
│   │           ├── all_MiniLM_L6_v2.pkl
│   │           ├── all_mpnet_base_v2.pkl
│   │           └── roberta.pkl
│   └── wikispeedia   				-> default directory to keep the original dataset
├── pip_requirements.txt
├── results.ipynb
├── results_semantic.ipynb			-> notebook demonstrating the semantic results
└── src						-> contains the source code
    └── semantic
        ├── clean_articles.py			-> script to clean HTML articles
        ├── generate_embeddings.py		-> script to generate embeddings
        ├── perform_clustering.py		-> script to perform clusterings
        └── utils				-> contains utility source codes
            ├── clustering_methods.py		-> source code of clustering methods
            ├── downloader.py			-> source code and script to download dataset
            ├── embedding_models.py		-> source code of embedding models
            └── evaluate_clustering.py		-> source code of clustering evaluation functions

```
