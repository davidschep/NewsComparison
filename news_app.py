###########################
#
# Main Streamlit application file
#
###########################


import os

import numpy as np
import pandas as pd
#import plotly.express as px

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

import news_connector
import webscrapingnews

###
# 1. Scrape Data
# 2. Select Datasets (all-the-news / scraped / ...)
# 3. Process Similar Articles
# 4. Article Statistics
###

# Application setup
st.set_page_config(page_title="News Comparison App", layout="wide")

# Title setup
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)
row0_1.title("News Comparison")
with row0_2:
    add_vertical_space()
row0_2.subheader(
    "Computational Tools for Data Science"
)

######### 1. Scrape Data #########
line1_spacer1, line1_1, line1_spacer2 = st.columns((0.1, 3.2, 0.1))
if 'select_agencies' not in st.session_state:
    st.session_state.select_agencies = []
if 'slider_articles_nr' not in st.session_state:
    st.session_state.slider_articles_nr = 0    

with line1_1:
    st.header("Optional: Scrape Data")#**{}**".format(user_name))
    st.markdown("_All agencies and 500 articles might take up to 20 minutes.._")
    select_agencies = st.multiselect("News Agencies", ['NY Post','Atlantic','CNN','Business Insider','Washington Post','Fox News','Guardian'])
    slider_articles_nr = st.slider("Number of Articles", 0, 500)
    if st.button("Scrape Web"):
        webscrapingnews.scraper(filename="new_scraping_data.csv", publication_list=select_agencies, max_limit_num_articles=int(slider_articles_nr))   
    
######### 2. Select Datasets #########
line2_spacer1, line2_1, line2_spacer2 = st.columns((0.1, 3.2, 0.1))
data_frames = []
if 'data' not in st.session_state:
    st.session_state.data = []
if 'selected_data' not in st.session_state:
    st.session_state.selected_data = []

with line2_1:
    st.header("Select Datasets")
    
    # select box for datasets
    st.session_state.datasets = [file for file in os.listdir("./data/") if file.endswith(('csv'))]
    if len(st.session_state.datasets) == 0:
        st.write("No datasets found in `data` folder.")
        st.stop()
    st.session_state.selected_data = st.multiselect("Select", st.session_state.datasets)

    # load data button
    if st.button("Load data"):
        for data_path in st.session_state.selected_data:
            data_frames.append(pd.read_csv(os.path.join('./data/', data_path)).head(500))
        st.session_state.data = pd.concat(data_frames)
        st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
        st.session_state.data = st.session_state.data.drop(['Unnamed: 0', 'year', 'month', 'url'], axis=1)
    
    if len(st.session_state.data) > 0:
        # display dataframe header
        st.write("First 20 entries in data:")
        st.dataframe(data=st.session_state.data.head(20))
        
        # display data statistics
        #..
    
    
######### 3. Process Similar Articles (News Connector) #########
line3_spacer1, line3_1, line3_spacer2 = st.columns((0.1, 3.2, 0.1))

with line3_1:
    if len(st.session_state.data) == 0:
        st.write("No datasets loaded.")
        st.stop()

    st.header("Process Similar Articles")
    
    # cluster articles button
    clustered_data = pd.DataFrame()
    if st.button("Cluster News Events"):
        # TODO: dynamically compute k
        st.session_state.data = news_connector.Cluster_Articles(k=13, df=st.session_state.data.copy())
        
    # display dataframe header
    st.write("First 20 entries in data with clusters:")
    st.dataframe(data=st.session_state.data.head(20))
    
    
    
######### 4. Article Statistics #########
line4_spacer1, line4_1, line4_spacer2 = st.columns((0.1, 3.2, 0.1))

with line4_1:
    if len(st.session_state.data) == 0:
        st.write("No datasets loaded.")
        st.stop()

    st.header("Analyze Articles")
    
    # TODO: six different methods or so for comparing articles
