import datetime
import re
import numpy as np
import pandas as pd
import os

# set random seed for dropping items
np.random.seed(10)

def load_data(path, data_limit):
    """
    Main function to load csv files, these files can be one of 3 options:
    1. Unfiltered data (dates and contents still need to be corrected)
    2. Filtered data
    3. Filtered data with clusters and event_id's

    Args:
        path (str): expected to be under 'data/' folder
        data_limit (int): max number of files loaded
        
    Returns:
        dataframe: ..
    """
    data = pd.read_csv(os.path.join('./data/', path))
    
    # drop indices randomly to reach data_limit size
    if data_limit < len(data):
        drop_indices = np.random.choice(data.index, len(data)-data_limit, replace=False)
        data = data.drop(drop_indices)
    
    # drop any unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # optional, filter data:
    # NOTE: we assume that data loaded is already filtered for now
    #data = filter_scraped_data(data)
    
    # old stuff:
    #data['date'] = pd.to_datetime(data['date'])
    #st.session_state.data = st.session_state.data.drop(['Unnamed: 0', 'year', 'month', 'url'], axis=1, errors='ignore')
    
    return data

def save_data(path, data):
    """
    Update CSV file (for instance when summaries are computed)

    Args:
        path (str): expected to be under 'data/' folder
        
    Returns:
        dataframe: ..
    """

    data.to_csv(os.path.join("data/", path))
    
def parse_ny_post_dates(df_row):
    date_string = df_row['date']
    date_string = date_string.replace('.m.', 'M').replace('ET', '').strip()
    date_format = "%b. %d, %Y, %I:%M %p"
    try:
        parsed_date = datetime.strptime(date_string, date_format)
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except ValueError:
        df_row['day'] = np.nan
        df_row['month'] = np.nan
        df_row['year'] = np.nan
    return df_row

def parse_guardian_dates(df_row):
    date_string = df_row['date']
    date_string = date_string.split(' ')[1:-1]
    date_string = ' '.join(date_string).replace('.', ':')

    date_format = "%d %b %Y %H:%M"

    try:
        parsed_date = datetime.strptime(date_string, date_format)
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except ValueError:
        df_row['day'] = np.nan
        df_row['month'] = np.nan
        df_row['year'] = np.nan
    
    return df_row

def parse_fox_news_dates(df_row):
    date_string = df_row['date']
    date_string = date_string.replace('EST', '').replace('pm', 'PM').replace('am', 'AM').strip()
    date_format_with_time = "%B %d, %Y %I:%M%p"
    date_format_without_time = "%B %d, %Y"
    try:
        parsed_date = datetime.strptime(date_string, date_format_with_time)
    except ValueError:
        try:
            parsed_date = datetime.strptime(date_string, date_format_without_time)
        except ValueError:
            df_row['day'] = np.nan
            df_row['month'] = np.nan
            df_row['year'] = np.nan
            return df_row
    df_row['day'] = parsed_date.day
    df_row['month'] = parsed_date.month
    df_row['year'] = parsed_date.year
    
    return df_row

def parse_atlantic_dates(df_row):
    date_string = df_row['date']
    try:
        parsed_date = datetime.fromisoformat(date_string.rstrip('Z'))
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except ValueError:
        df_row['day'] = np.nan
        df_row['month'] = np.nan
        df_row['year'] = np.nan
    
    return df_row

def parse_cnn_dates(df_row):
    date_string = df_row['date']
    date_string = date_string.replace(' EST', '').replace(' EDT', '').strip()
    date_format = "%I:%M %p, %a %B %d, %Y"

    try:
        parsed_date = datetime.strptime(date_string, date_format)
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except ValueError:
        df_row['day'] = np.nan
        df_row['month'] = np.nan
        df_row['year'] = np.nan
    
    return df_row

def parse_business_insider_dates(df_row):
    date_string = df_row['date']
    try:
        parsed_date = datetime.fromisoformat(date_string.rstrip('Z'))
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except ValueError:
        df_row['day'] = 'Unknown'
        df_row['month'] = 'Unknown'
        df_row['year'] = 'Unknown'
    
    return df_row

def parse_washington_post_dates(df_row):
    date_string = df_row['date']
    try:
        date_string = date_string.replace('at', '').replace('.m.', 'M').replace('EST', '').strip()
        date_format = "%B %d, %Y %I:%M %p"
        parsed_date = datetime.strptime(date_string, date_format)
        df_row['day'] = parsed_date.day
        df_row['month'] = parsed_date.month
        df_row['year'] = parsed_date.year
    except:
        df_row['day'] = np.nan
        df_row['month'] = np.nan
        df_row['year'] = np.nan
    return df_row

def clean_data(df_row):
    """
    Replaces reoccuring phrases in content which are not useful
    """
    try:
        df_row['content'] = df_row['content'].replace('Δ Thanks for contacting us. We\'ve received your submission.', '')
        df_row['content'] = df_row['content'].replace('Fox News Flash top headlines are here. Check out what\'s clicking on Foxnews.com', '')
    except:
        pass
    return df_row

def separate_joined_words(text):
    """
    Separates words like 'JUSTin' which have a bunch of capital letters followed by a small letter which is the start of 
    another word
    """
    try:
        pattern = re.compile(r'(?<=[a-z])(?=[A-Z])')
        separated_text = pattern.sub(' ', text)
        return separated_text
    except:
        return text
    
def filter_scraped_data(df1):
    """
    Takes unfiltered scraped dataframe as input and outputs filtered scraped dataframe
    """
    df1.drop(['summary'],axis=1,inplace=True)
    
    date_info_dict = {
            'NY Post': parse_ny_post_dates,
            'Atlantic': parse_atlantic_dates,
            'CNN': parse_cnn_dates,
            'Business Insider': parse_business_insider_dates,
            'Washington Post': parse_washington_post_dates,
            'Fox News': parse_fox_news_dates,
            'Guardian': parse_guardian_dates
        }
    
    df1['day'] = np.nan
    df1['month'] = np.nan
    df1['year'] = np.nan

    for name in date_info_dict:
        df1[df1['name'] == name] = df1[df1['name'] == name].apply(date_info_dict[name], axis=1)
    
    df1.dropna(subset=['content'], inplace=True)
    df1.reset_index(drop=True, inplace=True)
    
    df1 = df1.apply(clean_data, axis=1)
    
    df1['content'] = df1['content'].apply(separate_joined_words)
    df1['title'] = df1['title'].apply(separate_joined_words)
    
    return df1