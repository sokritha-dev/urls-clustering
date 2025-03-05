from urllib.parse import urlparse

def preprocess_url(url):
    parsed_url = urlparse(url)

    # Extract domain with subdomain (e.g., "example.com" from "news.example.com")
    domain_parts = parsed_url.netloc.split(".")
    domain = ".".join(domain_parts[-2:]) if len(domain_parts) > 2 else parsed_url.netloc

    # Extract path, replace "/" with spaces to make it word-based
    path = parsed_url.path.replace("/", " ").strip()

    # Extract query params (optional)
    query = parsed_url.query.replace("&", " ").replace("=", " ")

    # Combine extracted features
    processed_text = f"{domain} {path} {query}".strip()

    return processed_text



