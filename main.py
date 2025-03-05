import re
import sys
import time
from urllib.parse import urlparse
from algorithms.dbscan import dbscan
from algorithms.hierarchical_clustering import hierachical_clustering
from algorithms.k_mean import k_mean
from utils.load_storage import load_urls_from_csv, save_urls_to_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.scraping_urls import scrapefish_website_with_extract_rule


# Feature extraction function
def extract_url_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    path_tokens = re.split(r"[/-]", path)
    path_tokens = [token for token in path_tokens if token]
    return f"{domain} {' '.join(path_tokens)}"


def show_loading(message, duration=5):
    """Display a simple loading animation for the specified duration."""
    print(f"\n{message}", end="")
    for _ in range(duration):
        time.sleep(1)
        sys.stdout.write(".")
        sys.stdout.flush()
    print("\r" + " " * (len(message) + duration) + "\r")


def main():
    # Get user input for the website to scrape
    website = input(
        "Enter the website URL to scrape (e.g., https://www.realestate.com.kh/rent/): "
    )

    # Show loading animation while scraping URLs
    show_loading("Scraping URLs from the website...")

    # Scrape URLs from the website
    urls = scrapefish_website_with_extract_rule(website)

    # Save the URLs to CSV, overwriting existing content
    save_urls_to_csv("extracted_urls.csv", list(urls))

    # Load the URLs for clustering
    df = load_urls_from_csv("extracted_urls.csv")
    urls = df["url"].tolist()

    if not urls:  # Check if URLs were loaded
        print("No URLs found to cluster. Exiting.")
        return

    # Prepare features for clustering
    url_features = [extract_url_features(url) for url in urls]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(url_features)

    # Show loading animation while preparing features and clustering
    show_loading("Preparing features and clustering URLs...")

    # Run all clustering algorithms
    k_mean(df=df, X=X, length_url=len(urls))
    hierachical_clustering(df=df, X=X)
    dbscan(df=df, X=X)


if __name__ == "__main__":
    main()
