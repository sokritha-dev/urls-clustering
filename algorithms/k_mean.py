import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_mean(df, X, length_url):
    # Step 1: Calculate silhouette scores for different cluster numbers
    silhouette_scores = []
    cluster_range = range(
        2, min(11, length_url)
    )  # Start at 2 (silhouette needs >1 cluster)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"n_clusters={k}, Silhouette Score={score:.3f}")

    # Step 2: Pick the best n_clusters
    best_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBest number of clusters based on silhouette score: {best_n_clusters}")

    # Step 3: Cluster with the best n_clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df["cluster"] = labels

    # Step 4: Suggest groups to the user
    print("Suggested URL Groups:")
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
