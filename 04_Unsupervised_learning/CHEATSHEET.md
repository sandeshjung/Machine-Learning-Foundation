# Cheat Sheet: Unsupervised Learning

Notebooks: [kmeans_hierarchical](kmeans_hierarchical.ipynb) · [pca_tsne](pca_tsne.ipynb) · Theory: [README](README.md)

## Clustering
| | K-Means | Hierarchical (agglomerative) |
|---|---|---|
| Objective | Minimise WCSS $= \sum_k \sum_{x \in C_k} \lVert x - \mu_k \rVert^2$ | Repeatedly merge the two closest clusters |
| Algorithm | Assign each point to the nearest centroid, move centroids to cluster means, repeat | Build a dendrogram and cut it at a height or at $K$ clusters |
| Need $K$ up front? | Yes | No (choose it after looking at the dendrogram) |
| Cluster shape | Convex, similar-sized blobs | Depends on linkage |
| Cost | $O(nKd)$ per iteration | $O(n^2)$ memory, $O(n^2 \log n)$ time or worse |

**Linkage:** `ward` (minimises the variance increase, similar to K-Means) · `complete` (max distance, compact clusters) · `average` · `single` (min distance, produces chains)

**Choosing $K$**
- **Elbow:** plot WCSS against $K$ and look for the bend. WCSS always decreases, so don't just minimise it.
- **Silhouette:** $s = \frac{b - a}{\max(a, b)}$, where $a$ is the mean intra-cluster distance and $b$ the mean distance to the nearest other cluster. Ranges from $-1$ to $1$, and higher is better.
- **Comparing two clusterings:** the adjusted Rand index is invariant to label permutations (1 = identical partitions).

## Dimensionality reduction
| | PCA | t-SNE |
|---|---|---|
| Type | Linear projection | Non-linear embedding |
| Preserves | Global variance | Local neighbourhoods |
| Deterministic | Yes (up to the sign of each axis) | No (depends on the seed) |
| New data | `transform` works | No out-of-sample mapping |
| Use for | Compression, denoising, preprocessing, visualisation | **Visualisation only** |

**PCA via SVD:** centre $X$, then $X = U \Sigma V^\top$. Principal axes are the **rows of $V^\top$**, the projection is $X V_k$, explained variance is $\frac{\sigma_i^2}{n-1}$, and the ratio is $\frac{\sigma_i^2}{\sum_j \sigma_j^2}$. Keep enough components for about 90–95% cumulative variance.

**t-SNE knobs:** `perplexity` of 5–50 (the number of neighbours each point considers), `max_iter` ≥ 1000, `init="pca"` for stability. Reduce to about 50 dimensions with PCA first when $d$ is large.

## Pitfalls
- **Scale features.** Both K-Means and PCA are distance- or variance-based.
- K-Means is sensitive to initialisation. Use k-means++ and several restarts (`n_init`).
- Cluster labels are arbitrary integers, so compare clusterings with ARI or NMI, never label equality.
- In t-SNE plots, **cluster sizes and inter-cluster distances mean nothing**. Only "which points are near which" is meaningful.
- The sign of a PCA component is arbitrary, so align signs before comparing implementations.
- `torch.linalg.svd` returns $V^\top$, so transposing incorrectly silently gives wrong axes.
