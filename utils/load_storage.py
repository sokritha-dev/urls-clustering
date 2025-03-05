import pandas as pd


def save_urls_to_csv(filename, urls):
    """Save extracted URLs to a CSV file, overwriting any existing content."""
    try:
        # Convert the new URLs to a set to remove duplicates within the new scrape
        unique_urls = set(urls)

        # Convert to DataFrame and save to CSV, overwriting the existing file
        df = pd.DataFrame({"url": list(unique_urls)})
        df.to_csv(filename, index=False, encoding="utf-8")

        print(
            f"Overwrote {filename} with {len(unique_urls)} unique URLs from the new scrape."
        )
    except Exception as e:
        print(f"Error saving URLs: {e}")


def load_urls_from_csv(filename):
    """Load URLs from a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        return df  # Returns DataFrame with 'url' column
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return pd.DataFrame({"url": []})  # Return empty DataFrame if file doesn't exist
    except Exception as e:
        print(f"Error loading URLs: {e}")
        return pd.DataFrame({"url": []})
