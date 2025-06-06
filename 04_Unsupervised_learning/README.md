# Unsupervised Learning

**Unsupervised learning** is a branch of machine learning where algorithms learn patterns from data that has not been labeled, classified, or categorized. Unlike supervised learning, there are no explicit target outputs provided during training. The goal is to infer the natural structure present within the data.

## Clustering (K-Means & Hierarchical)

**Clustering** is a primary task in unsupervised learning. It involves partitioning a set of data points into a number of groups (clusters) such that:
*   Data points within the same cluster are highly similar to each other.
*   Data points in different clusters are dissimilar.
Similarity is typically measured using a distance metric (e.g., Euclidean distance).

### K-Means Clustering 

#### Concept and Goal

K-Means is a **partitioning clustering** algorithm. It aims to partition $N$ data points into $\large K$ distinct, non-overlapping clusters, where $\large K$ is a user-specified parameter. Each cluster is characterized by its **centroid** (mean of the points in that cluster). The goal is to minimize the **within-cluster sum of squares (WCSS)**, also known as inertia:

$$\large 
\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \mathbf{\mu}_k\|^2
$$

Where $\large C_k$ is the $\large k$-th cluster and $\large \mathbf{\mu}_k$ is its centroid.

#### Algorithm Steps
K-Means is an iterative algorithm:
1.  **Initialization:** Choose $\large K$ initial cluster centroids. This can be done by:
    *   Randomly selecting $\large K$ data points from the dataset.
    *   Using a more sophisticated method like K-Means++ (aims for better initial placement).
2.  **Assignment Step:** Assign each data point $\large \mathbf{x}_i$ to the cluster whose centroid $\large \mathbf{\mu}_k$ is closest (e.g., using Euclidean distance).

    $$\large 
    \text{label}(\mathbf{x}_i) = \arg\min_{k \in \{1,\dots,K\}} \|\mathbf{x}_i - \mathbf{\mu}_k\|^2
    $$

3.  **Update Step:** Recalculate the centroid of each cluster as the mean of all data points assigned to it in the previous step.
    $$\large 
    \mathbf{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
    $$

    Where $\large |C_k|$ is the number of points in cluster $\large C_k$.
4.  **Convergence:** Repeat steps 2 and 3 until the cluster assignments no longer change, the centroids no longer change significantly (below a tolerance), or a maximum number of iterations is reached.

<div align="center">
<img src="assets/kmeans.png">
<p>Fig.  K-Means Iteration - Showing points re-assigning and centroids moving</p>
</div>

#### Initialization of Centroids
*   **Random Initialization:** Simple but can sometimes lead to poor convergence or suboptimal clusters if initial centroids are poorly placed. Running K-Means multiple times with different random initializations and choosing the best result (lowest WCSS) is common.
*   **K-Means++:** A smarter initialization technique that aims to spread out the initial centroids, often leading to better and more consistent results.

#### Choosing the Number of Clusters (K)
Choosing the optimal $\large K$ is a common challenge. Two popular heuristic methods are:
*   **Elbow Method:**
    1.  Run K-Means for a range of $K$ values (e.g., $\large K=1$ to $\large 10$).
    2.  For each $\large K$, calculate the WCSS.
    3.  Plot WCSS against $\large K$.
    4.  Look for an "elbow" point in the plot: the point where adding another cluster does not significantly decrease WCSS. This point is often considered a good indication of $\large K$.
    
    <div align="center">
    <img src="assets/kmeans.png">
    <p>Fig. K-Means Elbow Method Plot - WCSS vs K</p>
    </div>

*   **Silhouette Analysis:**
    1.  For each data point, the silhouette coefficient measures how similar it is to its own cluster compared to other, neighboring clusters.
    2.  It ranges from -1 (poorly clustered) to +1 (densely clustered), with 0 indicating overlapping clusters.
    3.  Calculate the average silhouette score for different values of $K$. The $K$ that maximizes the average silhouette score is often chosen.

    <div align="center">
    <img src="assets/silhouette.png" width="900", height="350">
    <p>Fig. Silhouette Plot for K-Means</p>
    </div>

#### Advantages & Disadvantages
*   **Advantages:**
    *   Relatively simple to understand and implement.
    *   Computationally efficient for large datasets (scales well with the number of samples, often linear in $\large N$).
    *   Often converges quickly.
*   **Disadvantages:**
    *   Requires the number of clusters $\large K$ to be specified beforehand.
    *   Sensitive to the initial placement of centroids; can converge to local optima. (Run multiple times with different initializations).
    *   Assumes clusters are spherical, equally sized, and have similar densities, which may not hold for all datasets. Struggles with clusters of arbitrary shapes or varying densities.
    *   Sensitive to outliers, as they can pull centroids.

### Hierarchical Clustering

#### Concept (Agglomerative vs. Divisive)
Hierarchical clustering creates a hierarchy of clusters, often represented as a tree-like diagram called a **dendrogram**. There are two main approaches:
*   **Agglomerative (Bottom-Up):** Starts with each data point as its own individual cluster. In each step, the two closest clusters are merged until all points belong to a single cluster or a stopping criterion is met. This is the more common approach.
*   **Divisive (Top-Down):** Starts with all data points in a single cluster. In each step, a cluster is split into smaller clusters until each point is in its own cluster or a stopping criterion is met.

#### Agglomerative Algorithm Steps
1.  **Initialization:** Treat each data point as a singleton cluster.
2.  **Compute Proximity Matrix:** Calculate the pairwise distances (or similarities) between all initial clusters (points).
3.  **Merge Closest Clusters:** Find the two closest (most similar) clusters based on a chosen **linkage criterion** and merge them to form a new, larger cluster.
4.  **Update Proximity Matrix:** Update the distances between the new cluster and all other existing clusters.
5.  **Repeat:** Repeat steps 3 and 4 until all data points are merged into a single cluster or a desired number of clusters is achieved.

#### Linkage Criteria
The linkage criterion defines how the distance between two clusters (each potentially containing multiple points) is measured:
*   **Single Linkage:** The distance between two clusters is the minimum distance between any two points, one from each cluster. Can lead to "chaining" effect where clusters are elongated.
*   **Complete Linkage:** The distance between two clusters is the maximum distance between any two points, one from each cluster. Tends to produce more compact, spherical clusters.
*   **Average Linkage:** The distance between two clusters is the average distance between all pairs of points (one from each cluster). A compromise between single and complete linkage.
*   **Ward's Method:** Merges clusters such that the increase in the total within-cluster sum of squares (WCSS) is minimized. Tends to produce compact, roughly equal-sized clusters. Often a good default choice.

#### Dendrograms
A **dendrogram** is a tree diagram that visualizes the hierarchical clustering process.
*   The leaves of the tree are the individual data points.
*   Internal nodes represent merged clusters.
*   The height of the branches (or the y-axis) typically represents the distance (or dissimilarity) at which clusters were merged.
*   By "cutting" the dendrogram horizontally at a certain height (distance threshold), we can obtain a specific number of flat clusters.

<div align="center">
<img src="assets/dendograms.png">
<p>Fig. Example Dendrogram - Showing cluster merges and cutting</p>
</div>

#### Advantages & Disadvantages 
*   **Advantages:**
    *   Does not require the number of clusters $\large K$ to be specified beforehand; the dendrogram provides a view of clusterings at all levels of granularity.
    *   Can capture nested cluster structures.
    *   The dendrogram is a useful visualization of the data's structure.
*   **Disadvantages:**
    *   Computationally expensive, especially for large datasets (typically $\large O(N^2 \log N)$ or $\large O(N^3)$ for $\large N$ samples, depending on the implementation and linkage).
    *   The choice of linkage criterion and distance metric can significantly impact the results.
    *   Merges are irreversible; once a merge is made, it cannot be undone, which can lead to suboptimal clusterings if an early merge was poor.
    *   Can be sensitive to noise and outliers.
