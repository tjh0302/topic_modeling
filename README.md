# topic_modeling

## Uses a virtual environment with the updated requirements.txt 

## Probabilistic (Statistical) Approaches - Latent Dirichlet Allocation, LSA

Implements LDA for probabilistic topic modeling on a set of text. 
Functions are built out to visualize the model's topics, examine topics, identify the most probabilistic title, and map topics back to a pandas dataframe.

### Example Output from LDA 
<img width="1355" height="752" alt="image" src="https://github.com/user-attachments/assets/a205768e-3943-4af4-94e3-f76d9a7a079e" />


## Clustering Approaches: K-means, DBSCAN, HDBSCAN

### Implements various techniques for embeddings, dimensionality reduction, and visualization techniques
  - Vectorizer and Embeddings: TFIDF Vectorizer, transformer-based embeddings (code built but not implemented, need python version upgrade)
  - Dimensionality Reduction: PCA, t-SNE, UMAP
  - Model Evaluation: Silhouette Score
  - Topic Extraction: c-TF-IDF (built from scratch), BERTopic (not yet implemented)
  - Visualizations: 2D scatter plots of dimensionality-reduced clusters (colored), dendrograms (condensed, simple), word cloud (not yet implemented)

### Example Output from HDBSCAN + UMAP
<img width="1355" height="752" alt="image" src="https://github.com/user-attachments/assets/9073bc2d-838b-4928-9c01-84b2928a6768" />

### Optimization: grid search 

Iterates through 35,000 method and hyperparameter combinations and evaluates by finding the max() between each model's silhouette and DBCV score.
Greedy search on each vectorizer, dimension reduction method, clustering method, and clustering hyperparameters. 
(need to add in multiprocessing because this takes about 33 minutes)

#### Visualize grid search performance

<img width="851" height="697" alt="image" src="https://github.com/user-attachments/assets/3843ab1d-5b82-47fd-a3c9-d103ed9b6b44" />

### Cluster Label Assignment

Computes c-TF-IDF scores from scratch and maps labels back to the data. Visualizations not built (eventually need to build an hbar for each topic), althought the output for each topic with n_top_words is included.

## Example data - a small and a large dataset

Large (moderately large) - 2000 documents

Small - Created using ChatGPT my asking it to compile 100 sentences about finance, energy, and education.

## Environment
Utilized a virtual environment for dependencies which is represented by the requirements.txt

Requirements.txt needs updating from some packages installed in the notebook for clustering. These are reflected in the notebook.
