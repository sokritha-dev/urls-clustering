import os
from sklearn.cluster import DBSCAN


def dbscan(df, X):
    # Step 1: Apply DBSCAN
    dbscan = DBSCAN(
        eps=0.9, min_samples=10, metric="cosine"
    )  # Tune eps and min_samples as needed
    labels = dbscan.fit_predict(X.toarray())
    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )  # Remove noise (-1) from count
    print(
        f"Automatically determined number of clusters for DBSCAN: {n_clusters} (noise points marked as -1)"
    )

    # Step 2: Add cluster labels to the DataFrame
    # Replace -1 (noise) with a unique cluster ID for consistency (e.g., -1 → n_clusters)
    if -1 in labels:
        labels[labels == -1] = n_clusters  # Assign noise to a separate cluster
    df["cluster"] = labels

    # Step 3: Suggest groups to the user
    print("DBSCAN Suggested URL Groups:")
    for cluster_id in range(n_clusters + 1):  # Include noise cluster if present
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
    output_file = os.path.join(output_dir, "dbscan_clustering_results.csv")
    result_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nSaved DBSCAN clustering results to {output_file}")
