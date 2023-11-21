# NewsComparison
News Comparison project for Computational Tools for Data Science project.

### Setup
1. Install requirements: `pip install -r requirements.txt`.
2. Download [dataset](https://www.kaggle.com/datasets/snapcrack/all-the-news/) to `data/` folder (e.g. `data/articles1.csv`, so that program knows where to find files).
3. Download scraped data to `data` folder (`scraping_data.csv`).

### Run
Run `python -m streamlit run news_app.py --server.maxMessageSize 1000` to launch Streamlit program, or run individual notebooks for other tasks.

The `server.maxMessageSize 1000` parameter gives the program 1GB memory, which is needed for the 600MB+ dataset.
