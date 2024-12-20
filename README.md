# Emotion-Semantic Mapping through Wikipedia Article Transitions (M3 updated)

## Data Story
https://fht-epfl.github.io/adawebsite/

## Abstract

In today's digital age, where massive amounts of textual information are consumed, the interplay between semantics and emotions plays a critical role in shaping user experiences. This project examines the relationship between semantic transitions and emotional responses during navigation paths in the Wikispeedia game. By calculating semantic distances using text embeddings and analyzing emotion scores with a pre-trained emotion model, we explore how different types of semantic shifts evoke specific emotional responses. Through this, we aim to uncover patterns in emotional and cognitive engagement during navigation, contributing to user modeling and interaction design.



## Research Questions

1. How do semantic and emotional transitions interact during navigation paths?
2. what patterns emerge in the semantic-emotion mapping alignment or divergence across different path lengths?

## Methods

### Quantifying Semantic Changes
We use information-theoretic and embedding-based approaches to calculate semantic distances between articles. Two distance measures are employed:

- **Asymmetric Distance**: Reflects human-centric navigation choices based on Wikipedia transition data.
- **Symmetric Distance**: Balances bidirectional relationships, suitable for clustering and overall similarity analyses.

Text embeddings (MiniLM_L6_v2, MPNet) are processed using PCA to ensure dimensionality reduction while maintaining semantic integrity.


### Measuring Emotional Fluctuations

An emotion prediction model generates emotion scores per article. The emotional impact of navigation links is weighted based on:

- **Hyperlink Attention**: Focused on links most likely to guide users toward their target.

- **Sentence Weighting**: Assigns higher importance to linked content.
This allows emotion metrics to reflect nuanced user interactions.

### Correlation and Case Study

Semantic and emotional transitions are analyzed for correlations using statistical and clustering methods. Case studies (e.g., "Bird to Hitler") highlight the unique dynamics of surprise and curiosity metrics alongside semantic scores.

## Key Findings

- **Semantic Distance**: Longer paths reveal greater variability in semantic transitions, emphasizing exploratory navigation.
- **Emotional Variability**: Shorter paths exhibit higher emotional fluctuation, while longer paths show steadier emotional engagement.
- **Surprise and Curiosity**: Distinct behaviors emerge, with surprise linked to abrupt transitions and curiosity tied to smoother exploration.
For details, see the visualization and mapping in our Data Story linked above.

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
│       ├── semantic_paper_preprocessing.py  → Function for embedding (not calling for results.ipynb)
│       ├── semantic_paper_utils.py  → Utility functions for semantic analysis related to papers
│       ├── semantic_paper_visualization.py  → Visualization code for semantic paper analysis
│       ├── semantic_utils.py  → Utility functions for semantic analysis
│       └── semantic_visualization.py  → Code for visualizing semantic analysis results
├── .gitignore  → Specifies which files/folders Git should ignore
├── pip_requirements.txt  → List of Python packages required for the project
├── README.md  → Documentation for the project
└── results.ipynb  → Jupyter notebook for presenting and analyzing results



```

## Team Contribution (M3 updated)
- Ruyin Feng: Traditional Semantic Distance — exploring different embeddings, distance metrics, and clustering methods.
- Nastaran Hashemi: Paper Semantic Distance — evaluation method for semantic distance based on academic papers.
- Nathanael Lambert: Paper Semantic Distance — implementation of semantic distance methods derived from academic papers.
- Weilun Xu: Emotion Distance — definition and implementation; Data Story — content development.
- Haotian Fang: Correlation — analysis of the correlation between emotion and semantic distance; Data Story — frontend development.
