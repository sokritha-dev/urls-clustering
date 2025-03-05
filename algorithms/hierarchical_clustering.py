from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import os


def hierachical_clustering(df, X):
    """
    Perform Hierarchical (Agglomerative) clustering on URLs, suggest groups to the user,
    and save results to a CSV file in the 'outputs' folder.
    """
    # Step 1: Perform hierarchical clustering
    Z = linkage(X.toarray(), method="ward")  # 'ward' minimizes variance within clusters

    # Step 2: Automatically determine the number of clusters
    # Use the heights (distances) from the linkage matrix
    heights = Z[:, 2]  # Extract the distances (y-values) where clusters merge

    # Heuristic: Find the largest gap in heights to determine a natural cut
    # Sort heights in descending order
    sorted_heights = sorted(heights, reverse=True)
    gaps = [
        sorted_heights[i] - sorted_heights[i + 1]
        for i in range(len(sorted_heights) - 1)
    ]
    if gaps:  # Ensure there are gaps to analyze
        max_gap_idx = gaps.index(max(gaps))
        # Cut at the height just before the largest gap
        threshold = sorted_heights[max_gap_idx + 1]
    else:
        # Fallback: Use a simple rule (e.g., 2-5 clusters for small datasets)
        threshold = sorted_heights[0] * 0.7  # 70% of max height as a rough cut

    # Apply the threshold to get cluster labels
    labels = fcluster(Z, t=threshold, criterion="distance")
    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )  # Remove noise label if present
    print(f"Automatically determined number of clusters: {n_clusters}")

    # Step 3: Apply Agglomerative Clustering with the determined number (for consistency)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_final = clustering.fit_predict(X.toarray())
    df["cluster"] = labels_final

    # Step 4: Suggest groups to the user (keep printing for interaction)
    print("Suggested URL Groups:")
    for cluster_id in range(n_clusters):
        group_urls = df[df["cluster"] == cluster_id]["url"].tolist()
        print(f"\nGroup {cluster_id + 1} ({len(group_urls)} URLs):")
        for url in group_urls[:5]:
            print(f"  - {url}")
        if len(group_urls) > 5:
            print(f"  ... and {len(group_urls) - 5} more")

    # Step 5: Save results to a CSV file in the 'outputs' folder
    output_dir = "outputs"
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create 'outputs' folder if it doesn't exist

    # Create a new DataFrame with URLs and their cluster labels
    result_df = df[["url", "cluster"]].copy()
    output_file = os.path.join(output_dir, "hierarchical_clustering_results.csv")
    result_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\nSaved hierarchical clustering results to {output_file}")
