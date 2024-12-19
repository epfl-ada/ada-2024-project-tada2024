# Sentiment Analysis through User Navigation Paths (Wikispeedia)

## Abstract

This project explores the relationship between semantic changes and emotional fluctuations throughout human navigation paths on Wikispeedia. By combining semantic analysis techniques with emotional analysis based on a pre-trained emotion model, we aim to determine if and how certain semantic shifts induce specific types of emotions during the navigation process. The motivation is to understand the interplay between cognitive processing and emotional responses, offering insights into how people interact with information and potentially predicting emotional responses based on article structures.

## Research Questions

1. What is the relationship between semantic change and emotional fluctuation throughout human navigation paths?
2. Which types of emotions are induced by specific semantic jumps (e.g., significant shifts in topic or intensity)?
3. How does backtracking affect emotional progression in navigation paths?
4. Is there a correlation between semantic distance and specific emotional fluctuations such as curiosity or surprise?
5. Can we predict a user’s emotional response based on the sequence of articles they navigate through?

## Methods

### Experiment with Text Embeddings and Distance Measures

The first step is to find an alternative approach to represent the semantic distance between concepts. This step is crucial because the representation of semantic distance directly impacts the subsequent analyses. We utilize various text embeddings and distance measures to calculate differences between embeddings. The embedding models we have selected are:

- MiniLM_L6_v2
- mpnet_base_v2
- roberta

### Determining the Most Suitable Semantic Distance

We applied clustering to select the optimal semantic distance measure for this analysis, prioritizing the measure that results in minimal clustering bias as our semantic distance indicator. During the evaluation step, we first analyzed the consistency of the embedding results through ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information), and mapping clustering outputs to the distribution of article categories. From this mapping, we derived accuracy and F1 scores, which served as key metrics for identifying the most effective semantic distance measure. We also experimented with using PCA for data dimensionality reduction. Furthermore, we extracted meaning for K-Medoids by looking at the name and primary category of the center. All these results will serve as a foundation for selecting the most suitable semantic distance. The distance measures and corresponding clustering models we have selected are:

| Distance measure       | Corresponding clustering method |
|------------------------|---------------------------------|
| Euclidean Distance     | K-Means                         |
| Manhattan Distance     | K-Medoids                       |
| Cosine Distance        | K-Medoids                       |

In addition to the methods mentioned in this table, we also performed Spectral Clustering to enhance the credibility and validation of the evaluation results.

### Define and Calculate Emotion Scores

We used a benchmarked emotion prediction model to generate emotion scores, which output 26 types of emotion scores. Our approach considers human attention span and cognitive aspects of reading behavior:

- Users pay more attention to hyperlinks during navigation than to plain text.
- Attention increases when hyperlinks are more relevant to the target.
- Attention decays as reading progresses down the page, correlating with increasing reading sparsity.

Based on these observations, we weighted the emotion annotations from the pretrained model accordingly.

### Analyze Semantic Distance and Emotion Scores for Correlation

We performed an initial test to verify some degree of correlation between semantic distance and emotion scores. In the next phase, we will explore deeper relationships, such as correlation, causality, or induction, between each type of emotion and textual semantic change. Our final aim is to determine if certain semantic jumps (over α% distance change) induce specific types of emotions.

## Proposed Timeline

- **Milestone P1 (Week 10-11)**: Calculate more emotion scores, implement weighted attention scoring, and explore the impact of backtracking. Visualize initial findings on semantic distances and emotional responses.
- **Milestone P2 (Week 11-12)**: Perform correlation analysis, identify specific emotions linked to large semantic jumps, and finalize the interpretation of results.
- **Final Project (Week 13-14)**: Prepare the final report, and document all code.

## Organization within the Team

### Internal milestones:

- **Week 3**: Complete data preprocessing and embedding generation.
- **Week 4**: Finish clustering evaluation and choose semantic distance measure.
- **Week 6**: Complete weighted emotion calculations and backtracking analysis.
- **Week 8**: Finalize correlation analysis and prepare findings for milestone 2.


## Questions for TAs
Would it be appropriate to use additional emotional models to validate the findings, or should we stick with one benchmarked model to maintain consistency?


## Important Download this and unzip it to replace "data" folder
https://drive.google.com/file/d/1yPs-of3ya39vxxoNtSTrrxHPKYBDlmfY/view?usp=drive_link

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

The directory structure of the project ():

```
ADA-2024-PROJECT-TADA2024/
├── data/ (only important files of data are shown below)
│   ├── emotion/ 			
│   ├── original_dataset/			->
│   └── semantic/			->
├── src/
│   ├── correlation/
│   │   └── correlation_utils.py			->
│   ├── emotion/
│   │   ├── emotion_scores.py
│   │   ├── emotion_utils.py
│   │   └── emotion_visualization.py
│   └── semantic/
│       ├── semantic_paper_utils.py
│       ├── semantic_paper_visualization.py
│       ├── semantic_utils.py
│       └── semantic_visualization.py
├── .gitignore
├── pip_requirements.txt
├── README.md
└── results.ipynb

```
