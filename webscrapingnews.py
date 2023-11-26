#!/usr/bin/env python
# coding: utf-8

# Import relevant libraries
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time


## Scraper function to scrape articles from The Guardian
def scrape_guardian_article(df_row):
    # Get soup
    soup = df_row['soup']

    # Attempt to get summary
    summary_tag = soup.find('div', {'data-gu-name': 'standfirst'})
    summary = summary_tag.get_text(strip=True) if summary_tag else None

    # Attempt to get author
    author_tag = soup.find('a', rel='author')
    author = author_tag.get_text(strip=True) if author_tag else 'Unknown'

    # Attempt to get content
    content_tags = soup.find_all('p')
    content = ' '.join(tag.get_text(strip=True) for tag in content_tags[1:])

    # Attempt to get date
    date_tag = soup.find('span', class_='dcr-u0h1qy')
    date_string = date_tag.get_text(strip=True) if date_tag else 'Unknown'

    """
    # Parse the date if it's not 'Unknown'
    if date_string != 'Unknown':
        try:
            date_format = "%B %d, %Y %I:%M%p"
            date = datetime.strptime(date_string, date_format)
        except ValueError:
            date = 'Unknown'
    else:
        date = 'Unknown'

    """

    # Update the df_row with the scraped data
    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    #df_row['date'] = date
    df_row['date'] = date_string

    return df_row

## Scraper function to scrape articles from The Washington Post
def scrape_washington_post_article(df_row):
    # Get soup
    soup = df_row['soup']

    # Attempt to get summary
    summary = None
    summary_candidates = ['PJLV PJLV-iPJLV-css grid-center w-100']
    for candidate in summary_candidates:
        summary_tag = soup.find(class_=candidate)
        if summary_tag:
            summary = summary_tag.get_text(strip=True)
            break

    # Attempt to get author
    author = None
    author_candidates = ['wpds-c-cNdzuP wpds-c-cNdzuP-ejzZdU-isLink-true', 'a[data-qa="author-name"]']
    authors = []
    for candidate in author_candidates:
        author_tags = soup.find_all(class_=candidate) or soup.find_all('a', {'data-qa': 'author-name'})
        for tag in author_tags:
            authors.append(tag.get_text(strip=True))
    author = ' and '.join(authors)

    # Get content
    content_list = soup.find_all(class_='article-body grid-center grid-body') or soup.find_all('p')
    content = ' '.join(content.get_text(strip=True) for content in content_list)

    # Get date
    date_string = None
    date_candidates = ['wpds-c-iKQyrV', 'wpds-c-iKQyrV wpds-c-iKQyrV-ihqANPJ-css overrideStyles']
    for candidate in date_candidates:
        date_tag = soup.find(class_=candidate)
        if date_tag:
            date_string = date_tag.get_text(strip=True)
            #date_string = date_string.replace('.m.', 'm').replace('EST', '').strip()
            break

    # Parse the date
    #date_format = "%B %d, %Y at %I:%M %p"
    #date = datetime.strptime(date_string, date_format) if date_string else None

    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    #df_row['date'] = date
    df_row['date'] = date_string

    return df_row

## Scraper function to scrape articles from The New York Post
def scrape_ny_post_article(df_row):
    # Get soup
    soup = df_row['soup']

    # No summaries exist
    summary = None

    # Get author
    author_tag = soup.find('span', class_='meta__link')
    author = author_tag.get_text(strip=True)[16:] if author_tag else 'Unknown'

    # Get content
    content_list = soup.find_all('p')[1:-1]
    content = ' '.join(content.get_text(strip=True) for content in content_list)

    # Get date
    date_tag = soup.find('div', class_='date--updated__item')
    date_string = date_tag.find_all('span')[1].get_text(strip=True) if date_tag else 'Unknown'
    #date_string = date_string.replace('.m.', 'M').replace('ET', '').strip()
    #date_format = "%b. %d, %Y, %I:%M %p"
    #try:
        #date = datetime.strptime(date_string, date_format)
    #except ValueError:
        #date = 'Unknown'

    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    #df_row['date'] = date
    df_row['date'] = date_string

    return df_row

## Scraper function to scrape articles from The Atlantic
def scrape_atlantic_article(df_row):
    # Get soup
    soup = df_row['soup']

    # Attempt to get summary
    summary_tag = soup.find(class_='ArticleHero_dek__EqdkK')
    summary = summary_tag.get_text(strip=True) if summary_tag else None

    # Attempt to get author
    author_tag = soup.find(class_='ArticleBylines_link__kNP4C')
    author = author_tag.get_text(strip=True) if author_tag else 'Unknown'

    # Attempt to get content
    content_tags = soup.find_all('p', class_='ArticleParagraph_root__4mszW')
    content = ' '.join(tag.get_text(strip=True) for tag in content_tags)

    # Attempt to get date
    date_tag = soup.find('time', class_='ArticleTimestamp_root__b3bL6')
    date = date_tag['datetime'] if date_tag else 'Unknown'

    # Parse the date if it's not 'Unknown'
    #if date != 'Unknown':
        #try:
            #date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ').isoformat()
        #except ValueError:
            #date = 'Unknown'

    # Update the df_row with the scraped data
    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    df_row['date'] = date

    return df_row

## Scraper function to scrape articles from CNN
def scrape_cnn_article(df_row):
    # Get soup
    soup = df_row['soup']

    # No summaries for CNN articles
    summary = None

    # Check if it's a live news article and get the author, content, and date accordingly
    if '/live-news/' in df_row['url']:
        author_tag = soup.find('p', {'data-type': 'byline-area'})
        author = author_tag.get_text(strip=True) if author_tag else 'Unknown'

        content_tags = soup.find_all('p', class_='sc-gZMcBi render-stellar-contentstyles__Paragraph-sc-9v7nwy-2 dCwndB')
        content = ' '.join(tag.get_text(strip=True) for tag in content_tags)

        date_tag = soup.find('div', class_='hJIoKL')
        date = date_tag.get_text(strip=True) if date_tag else 'Unknown'
        date_string = ' '.join(date.split(' ')[-3:])+' '+str(date.split(' ')[2][1:3])+':'+(date.split(' ')[2][3:5]) if date!='Unknown' else 'Unknown'
    else:
        author_tag = soup.find(class_='byline__link')
        author = author_tag.get_text(strip=True) if author_tag else 'Unknown'

        content_tags = soup.find_all('p', class_='paragraph inline-placeholder')
        content = ' '.join(tag.get_text(strip=True) for tag in content_tags)

        date_tag = soup.find(class_='timestamp')
        #date_string = ' '.join(date_tag.get_text(strip=True).split(' ')[-7:][-3:]+date_tag.get_text(strip=True).split(' ')[-7:][0:2]) if date_tag else 'Unknown'
        date_string = date_tag.get_text(strip=True).split('\n')[-1] if date_tag else 'Unknown'

    # Parse the date if it's not 'Unknown'
    """
    if date_string != 'Unknown':
        try:
            if '/live-news/' in df_row['url']:
                date_format = "%B %d, %Y %H:%M"
            else:
                date_format = '%B %d, %Y %I:%M %p'
            date = datetime.strptime(date_string, date_format).isoformat()
        except ValueError:
            date = 'Unknown'
    """

    # Update the df_row with the scraped data
    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    #df_row['date'] = date
    df_row['date'] = date_string

    return df_row

## Scraper function to scrape articles from Business Insider
def scrape_business_insider_article(df_row):
    # Get soup
    soup = df_row['soup']
    
    # No summary for BI
    summary = None

    # Attempt to get author
    author_tag = soup.find(class_='byline-author headline-bold')
    author = author_tag.get_text(strip=True) if author_tag else 'Unknown'

    # Attempt to get content
    content_tags = soup.find_all('p')
    content = ' '.join(tag.get_text(strip=True) for tag in content_tags[1:])

    # Attempt to get date
    date_tag = soup.find('div', class_='byline-timestamp')
    date = date_tag['data-timestamp'] if date_tag else 'Unknown'

    # Update the df_row with the scraped data
    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    df_row['date'] = date

    return df_row

## Scraper function to scrape articles from Fox News
def scrape_fox_news_article(df_row):
    # Get soup
    soup = df_row['soup']

    # Attempt to get summary
    summary_tag = soup.find('h2', class_='sub-headline speakable')
    summary = summary_tag.get_text(strip=True) if summary_tag else None

    # Attempt to get author
    author_tag = soup.find(class_='author-byline')
    author = author_tag.get_text().split('\n')[-1].replace('Fox News','').strip() if author_tag else 'Unknown'

    # Attempt to get content
    content_tags = soup.find_all('p')
    content = ' '.join(tag.get_text(strip=True) for tag in content_tags[1:])

    # Attempt to get date
    date_tag = soup.find('time')
    #date_string = date_tag.get_text(strip=True)[:-4] if date_tag else 'Unknown'
    date_string = date_tag.get_text(strip=True) if date_tag else 'Unknown'

    """
    # Parse the date if it's not 'Unknown'
    if date_string != 'Unknown':
        try:
            date_format = "%B %d, %Y %I:%M%p"
            date = datetime.strptime(date_string, date_format)
        except ValueError:
            date = 'Unknown'
    else:
        date = 'Unknown'

    """

    # Update the df_row with the scraped data
    df_row['summary'] = summary
    df_row['author'] = author
    df_row['content'] = content
    #df_row['date'] = date
    df_row['date'] = date_string

    return df_row

# Scrape Articles from Top News pages

class Top_News:
    # Class initializer with publication names and setup for scraping functions and class data
    def __init__(self, publication_names):
        self.publication_names = publication_names if publication_names != 'all' else ['NY Post', 'Atlantic', 'CNN', 'Business Insider', 'Washington Post', 'Fox News', 'Guardian']
        self.results_df = pd.DataFrame()
        # Dictionary containing scraping function for each publication
        self.scrapers = {
            'NY Post': scrape_ny_post_article,
            'Atlantic': scrape_atlantic_article,
            'CNN': scrape_cnn_article,
            'Business Insider': scrape_business_insider_article,
            'Washington Post': scrape_washington_post_article,
            'Fox News': scrape_fox_news_article,
            'Guardian': scrape_guardian_article
        }
        # Dictionary containing class data for each publication to locate articles on their homepage
        self.class_data = {
            'NY Post': ['https://nypost.com/', 'story__headline headline headline--xl', 'story__headline headline headline--sm', 'story__headline headline headline--combo-lg-xl headline--with-inline-webwood'],
            'Atlantic': ['https://www.theatlantic.com/world/', 'HomepageBottom_channelArticle__2wxRe', 'SmallPromoItem_root__nkm_2', 'Lede_title__7Wg1g', 'Offlede_title__kiinC', 'QuadBelt_title__mB6Zf', 'DoubleWide_title__diUPi', 'DoubleStack_title___FhPb', 'Latest_article__DW75m', 'Popular_listItem__CtMMj'],
            'CNN': ['https://edition.cnn.com/', 'container__link', 'container__title container_lead-package__title container__title--emphatic hover container__title--emphatic-size-l1'],
            'Business Insider': ['https://www.businessinsider.com/', 'tout', 'quick-link', 'most-popular-item', '.featured-tout-collection-wrapper .tout-title a', '.two-column-wrapper .tout-title-link'],
            'Washington Post': ['https://www.washingtonpost.com/', 'wpds-c-iiQaMf wpds-c-iiQaMf-igUpeQR-css', 'wpds-c-iiQaMf wpds-c-iiQaMf-ikZTsyd-css', 'wpds-c-iiQaMf wpds-c-iiQaMf-ibYgSwf-css'],
            'Fox News': ['https://www.foxnews.com/', 'article'],
            'Guardian': ['https://www.theguardian.com/', 'dcr-12ilguo', 'dcr-yw3hn9']
        }
    
    def _get_soup(self, url,rate_limit_seconds=1):
        try:
            # implement rate limiting for ethical reasons
            time.sleep(rate_limit_seconds)
            # try to get response from server and parse it into a BeautifulSoup object
            response = requests.get(url)
            response.raise_for_status()  
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            # handle exceptions
            print(f"Request failed for {url}: {e}")
            return None

    def scrape_publications(self):
        # use class data to find and parse articles from homepages
        selected_class_data = {name: self.class_data[name] for name in self.publication_names if name in self.class_data}
        articles_data = []
           
        for class_name in selected_class_data:
            # find articles and collect data
            base_url = selected_class_data[class_name][0]
            soup = self._get_soup(base_url)
            
            # find article blocks
            article_blocks = []
            for c_ in selected_class_data[class_name][1:]:
                temp = soup.find_all(class_=c_) if (class_name != 'Fox News') else soup.find_all('article')
                temp = temp if temp else soup.select(c_)
                article_blocks.extend(temp)
            
            # extract data from each block
            for block in article_blocks:
                url = block.find('a')['href'] if block.find('a') else (block['href'] if 'href' in block.attrs else None)
                if url:
                    article_data = {
                        'name': class_name,
                        'title': block.get_text(strip=True),
                        'url': base_url[:-1]+url if url.startswith('/') else url,
                        'soup': self._get_soup(url) # get content of article
                    }
                    articles_data.append(article_data)

        self.results_df = pd.DataFrame(articles_data)
        # drop duplicates
        self.results_df.drop_duplicates(subset=['url'], inplace=True)
        self.results_df.reset_index(drop=True, inplace=True)

        return self.results_df

    def article_distribution(self):
        if not self.results_df.empty:
            # plot distribution of articles from different publishers
            self.results_df.groupby('name').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
            plt.gca().spines[['top', 'right',]].set_visible(False)
            plt.show()
        else:
            print('Scrape publications first')

    def scrape_articles(self, max_articles_per_publication=None):
        if self.results_df.empty:
            print("Scraping publications for article URLs...")
            self.scrape_publications()

        all_article_data = []
        articles_count = {}

        for index, row in self.results_df.iterrows():
            publication_name = row['name']
            if publication_name in self.scrapers:
                articles_count.setdefault(publication_name, 0)
                # do not scrape more than the desired number of articles per publisher
                if max_articles_per_publication is None or articles_count[publication_name] < max_articles_per_publication:
                    try:
                        article_data = self.scrapers[publication_name](row)
                        all_article_data.append(article_data)
                        articles_count[publication_name] += 1
                    except Exception as e:
                        print(f"An error occurred while scraping {row['url']}: {e}")

        self.results_df = pd.DataFrame(all_article_data)
        self.results_df.reset_index(drop=True, inplace=True)

def scraper(filename='scraping_data.csv', publication_list='all', max_limit_num_articles=None):
    news_scraper = Top_News(publication_list)
    news_scraper.scrape_publications()
    #news_scraper.article_distribution()
    news_scraper.scrape_articles(max_articles_per_publication=max_limit_num_articles)
    filename = os.path.join('./data/', filename)
    news_scraper.results_df.to_csv(filename, index=False)