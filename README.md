# topic_modeling

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

## Example data - a small and a large dataset

Large (moderately large) - 2000 documents

Small - Created using ChatGPT my asking it to compile 100 sentences about finance, energy, and education.

## Environment
Utilized a virtual environment for dependencies which is represented by the requirements.txt

Requirements.txt needs updating from some packages installed in the notebook for clustering. These are reflected in the notebook.
