import os
from dotenv import load_dotenv

load_dotenv()

# Get API credentials from environment variables
SCRAPE_FISH_API_KEY = os.getenv("SCRAPE_FISH_API_KEY")
SCRAPE_FISH_URI = os.getenv("SCRAPE_FISH_URI", "https://scrapingfish.com/api")
