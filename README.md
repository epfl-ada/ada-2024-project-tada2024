# Emotion-Semantic Mapping through Wikipedia Article Transitions (M3 updated)

## Team Contribution (M3 updated)
- Ruyin Feng (Traditional Semantic Distance)
- Nastaran Hashemi (Paper Semantic Distance)
- Nathanael Lambert (Paper Semantic Distance)
- Weilun Xu (Emotion Distance, Data Story)
- Haotian Fang (Correlation, Data Story)


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


### Determining the Most Suitable Semantic Distance

We applied clustering to select the optimal semantic distance measure for this analysis, prioritizing the measure that results in minimal clustering bias as our semantic distance indicator. During the evaluation step, we first analyzed the consistency of the embedding results through ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information), and mapping clustering outputs to the distribution of article categories. From this mapping, we derived accuracy and F1 scores, which served as key metrics for identifying the most effective semantic distance measure. We also experimented with using PCA for data dimensionality reduction. Furthermore, we extracted meaning for K-Medoids by looking at the name and primary category of the center. All these results will serve as a foundation for selecting the most suitable semantic distance. The distance measures and corresponding clustering models we have selected are:

| Distance measure       | Corresponding clustering method |
|------------------------|---------------------------------|
| Euclidean Distance     | K-Means                         |
| Manhattan Distance     | K-Medoids                       |
| Cosine Distance        | K-Medoids                       |

In addition to the methods mentioned in this table, we also performed Spectral Clustering to enhance the credibility and validation of the evaluation results.

### Define and Calculate Emotion Scores

We used a benchmarked emotion prediction model to generate emotion scores, which output 28 types of emotion scores. Our approach considers human attention span and cognitive aspects of reading behavior:

- Users pay more attention to hyperlinks during navigation than to plain text.
- Attention increases when hyperlinks are more relevant to the target.
- Attention decays as reading progresses down the page, correlating with increasing reading sparsity.

Based on these observations, we weighted the emotion annotations from the pretrained model accordingly.

### Analyze Semantic Distance and Emotion Scores for Correlation

We performed an initial test to verify some degree of correlation between semantic distance and emotion scores. In the next phase, we will explore deeper relationships, such as correlation, causality, or induction, between each type of emotion and textual semantic change. Our final aim is to determine if certain semantic jumps (over α% distance change) induce specific types of emotions.


## Organization within the Team

### Internal milestones: (M3 Updated)

- **Week 3**: Complete data preprocessing and embedding generation.
- **Week 4**: Finish clustering evaluation and choose semantic distance measure.
- **Week 6**: Complete weighted emotion calculations and backtracking analysis.
- **Week 8**: Finalize correlation analysis and prepare findings for milestone 2.
- **Week 10**: Implement semantic distance in the paper and improve clustering evaluation.
- **Week 11**: Finish correlating emotion and semantics.
- **Week 12**: Conduct case study.
- **Week 13**: Draft the first version of the data story.
- **Week 14**: Finalize the data story, results, and GitHub documentation.


## Important Download this and unzip it to replace "data" folder
https://drive.google.com/file/d/1zSDj7c8xEAkieHe9v-eccjGk-43NmfTw/view?usp=sharing

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

## Project Structure (M3 Updated)

The directory structure of the project:

```
ADA-2024-PROJECT-TADA2024/
├── data/  → the following subfolders are empty, please find the google drive link below to replace the whole data folder
│   ├── correlation/  → Folder containing correlation-related data files
│   ├── emotion/  → Folder containing emotion-related data files
│   ├── original_dataset/  → Folder containing the original dataset
│   └── semantic/  → Folder containing semantic-related data files
├── src/
│   ├── correlation/
│   │   ├── correlation_utils.py  → Utility functions for calculating correlations
│   │   └── correlation_visualization.py  → Code for visualizing correlation results
│   ├── emotion/
│   │   ├── emotion_scores.py  → Script for scoring emotions based on the dataset
│   │   ├── emotion_utils.py  → Utility functions for emotion-related analysis
│   │   └── emotion_visualization.py  → Code for visualizing emotion-related results
│   └── semantic/
│       ├── semantic_paper_utils.py  → Utility functions for semantic analysis related to papers
│       ├── semantic_paper_visualization.py  → Visualization code for semantic paper analysis
│       ├── semantic_utils.py  → Utility functions for semantic analysis
│       └── semantic_visualization.py  → Code for visualizing semantic analysis results
├── .gitignore  → Specifies which files/folders Git should ignore
├── pip_requirements.txt  → List of Python packages required for the project
├── README.md  → Documentation for the project
└── results.ipynb  → Jupyter notebook for presenting and analyzing results



```
