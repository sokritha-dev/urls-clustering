# URL Scraper and Clustering Tool

This project is a Python tool for scraping URLs from a website, clustering them into meaningful groups based on their structure, and saving the results for analysis. It uses machine learning algorithms (KMeans, Hierarchical Clustering, and DBSCAN) to categorize URLs, making it easy for users to select groups for further processing or crawling.

## Overview

The tool consists of a single script (`main.py`) that:
- Accepts a website URL as user input in the terminal.
- Scrapes URLs from the specified website using a custom scraping utility.
- Saves the scraped URLs to `extracted_urls.csv`.
- Clusters the URLs using three algorithms: KMeans, Hierarchical Clustering, and DBSCAN.
- Displays clustering results, allows users to select a group to crawl, and saves detailed results to the `outputs` folder.

## Features
- **Dynamic URL Scraping**: Scrape URLs from any website provided by the user.
- **Automatic Clustering**: Use KMeans, Hierarchical Clustering, or DBSCAN to group URLs without manual configuration.
- **Loading Feedback**: Show loading animations during scraping and clustering for a better user experience.
- **Result Storage**: Save clustering results to CSV files in the `outputs` folder, named by algorithm (e.g., `kmeans_clustering_results.csv`).
- **Error Handling**: Robust handling of file operations, scraping errors, and clustering issues.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Install Dependencies
1. Navigate to the project directory.
2. Create a virtual environment in Python:
    ```bash
    python -m venv venv
3. Activate the virtual environment:
    - On Windows:
    ```bash
    "venv\Scripts\activate"
    - On macOS/Linux:
    "source venv/bin/activate"
3. Install the required packages using:
   ```bash
   pip install -r requirements.txt

### Project Structure
```bash
project_root/
├── algorithms/           # (Optional, if you keep separate clustering files)
│   ├── k_mean.py
│   ├── hierarchical_clustering.py
│   └── dbscan.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── load_storage.py   # Functions for saving/loading CSV files
│   └── scraping_urls.py  # Functions for scraping URLs
├── outputs/              # Folder for clustering results (auto-created)
│   ├── kmeans_clustering_results.csv
│   ├── hierarchical_clustering_results.csv
│   └── dbscan_clustering_results.csv
├── extracted_urls.csv    # Stores scraped URLs
├── main.py               # Main script for scraping and clustering
├── .gitignore            # Git ignore file
├── .env                  # Environment variables (optional)
├── .env.sample           # Sample Environment variables (optional)
├── load_env.py           # Environment loading script (optional)
└── README.md             # This file
```

### Usage
## With Scrapefish API (Scraping Enabled)
If you have access to the Scrapefish API and want to scrape URLs:
1. Create a .env file in the project root, following the format in .env.sample
2. Run the main script from the terminal:
   ```bash
   python main.py
3. When prompted, enter the website URL you want to scrape (e.g., https://www.realestate.com.kh/rent/, etc.).
4. Review the clustering results for KMeans, Hierarchical Clustering, and DBSCAN, which will be printed to the terminal.
5. Check the outputs folder for CSV files containing the detailed clustering results (e.g., kmeans_clustering_results.csv).

## Without Scrapefish API (Scraping Enabled)
If you don’t have the Scrapefish API, you must provide an extracted_urls.csv file with a list of URLs to cluster. Otherwise, clustering won’t work:
1. Prepare an extracted_urls.csv file in the project root with a column named url containing the list of URLs (e.g., /jobs/1049, https://www.facebook.com/jobify.works/).
2. Comment out some line of code in the main.py that used for asking to input the website url and save to extracted_urls.csv
3. Run the main script from the terminal:
   ```bash
   python main.py
4. When prompted, enter the website URL you want to scrape (e.g., https://www.realestate.com.kh/rent/, etc.).
5. Review the clustering results for KMeans, Hierarchical Clustering, and DBSCAN, which will be printed to the terminal.
6. Check the outputs folder for CSV files containing the detailed clustering results (e.g., kmeans_clustering_results.csv).