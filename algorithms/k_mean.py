import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def k_mean(df, X, length_url):
    """
    Perform KMeans clustering with Elbow Method, Silhouette Score analysis, and visualize results.
    """
    cluster_range = range(2, min(11, length_url))  # K from 2 to 10 (or length_url)
    wcss = []  # For Elbow Method
    silhouette_scores = []  # For Silhouette Score

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        labels = kmeans.fit_predict(X.toarray())

        # Compute WCSS (Elbow Method)
        wcss.append(kmeans.inertia_)

        # Compute Silhouette Score
        score = silhouette_score(X.toarray(), labels)
        silhouette_scores.append(score)
        print(f"n_clusters={k}, Silhouette Score={score:.3f}")

    # Find best K using Silhouette Score
    best_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBest number of clusters based on silhouette score: {best_n_clusters}")

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, wcss, marker="o", linestyle="-", color="b")
    plt.title("Elbow Method: Finding Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.grid(True)
    plt.axvline(
        x=best_n_clusters,
        color="r",
        linestyle="--",
        label=f"Best K = {best_n_clusters}",
    )
    plt.legend()
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, silhouette_scores, marker="o", linestyle="-", color="g")
    plt.title("Silhouette Scores for Different K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.axvline(
        x=best_n_clusters,
        color="r",
        linestyle="--",
        label=f"Best K = {best_n_clusters}",
    )
    plt.legend()
    plt.show()

    # Run KMeans with the best K
    kmeans = KMeans(n_clusters=best_n_clusters, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(X.toarray())
    df["cluster"] = labels

    # Visualize Clusters using t-SNE
    print("\nVisualizing Clusters in 2D...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X.toarray())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette="viridis", alpha=0.7
    )
    plt.title("t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()

    # Print Suggested URL Groups
    print("\nSuggested URL Groups:")
    for cluster_id in range(best_n_clusters):
        group_urls = df[df["cluster"] == cluster_id]["url"].tolist()
        print(f"\nGroup {cluster_id + 1} ({len(group_urls)} URLs):")
        for url in group_urls[:5]:
            print(f"  - {url}")
        if len(group_urls) > 5:
            print(f"  ... and {len(group_urls) - 5} more")

    # Save results
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    result_df = df[["url", "cluster"]].copy()
    output_file = os.path.join(output_dir, "kmeans_clustering_results.csv")
    result_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nSaved KMeans clustering results to {output_file}")
