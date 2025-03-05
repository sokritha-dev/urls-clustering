import json
import logging
import requests

from load_env import SCRAPE_FISH_API_KEY, SCRAPE_FISH_URI


def scrapefish_website_with_extract_rule(website_url: str) -> list:
    """Extract unique links from the target website using Scraping Fish API."""
    try:
        payload = {
            "api_key": SCRAPE_FISH_API_KEY,
            "url": website_url,
            "render_js": "true",
            "extract_rules": json.dumps(
                {"links": {"type": "all", "selector": "a", "output": "@href"}}
            ),
        }

        response = requests.get(SCRAPE_FISH_URI, params=payload)
        response_json = response.json()

        # Safely handle missing 'links'
        items = response_json.get("links", [])
        if not items:
            logging.warning(f"No 'links' found in response for {website_url}")

        # Remove duplicates by converting list to a set and back
        unique_items = list(set(items))

        return unique_items
    except Exception as e:
        logging.error(f"Scraping failed for {website_url}: {e}")
        return []  # Return empty list on failure
