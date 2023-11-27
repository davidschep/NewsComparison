###########################
#
# Main Streamlit application file
#
###########################


import os

import numpy as np
import pandas as pd
import plotly.express as px

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
    slider_articles_nr = st.number_input("Number of Articles", 0, 500, value=10)
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
        st.warning("No datasets found in `data` folder.")
        st.stop()
    st.session_state.selected_data = st.multiselect("Select", st.session_state.datasets)

    # load data button
    if st.button("Load data"):
        for data_path in st.session_state.selected_data:
            DATA_LIMIT = 520 # how many items to load
            data_frames.append(pd.read_csv(os.path.join('./data/', data_path)).head(DATA_LIMIT))
        st.session_state.data = pd.concat(data_frames)
        st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
        st.session_state.data = st.session_state.data.drop(['Unnamed: 0', 'year', 'month', 'url'], axis=1, errors='ignore')
    
    if len(st.session_state.data) > 0:
        # display dataframe header
        st.write("First 20 entries in data (size " + str(len(st.session_state.data)) + "):")
        st.dataframe(data=st.session_state.data.head(20))
        
        # display data statistics
        #..
    
    
######### 3. Process Similar Articles (News Connector) #########
line3_spacer1, line3_1, line3_spacer2 = st.columns((0.1, 3.2, 0.1))

with line3_1:
    if len(st.session_state.data) == 0:
        st.warning("No datasets loaded.")
        st.stop()

    st.header("Process Similar Articles")
    st.markdown("Clustering of articles takes a couple minutes for 500 articles")
    
    # nr clusters
    nr_clusters = st.number_input("Number of clusters", 0, 50, value=15)
    
    # cluster articles button
    clustered_data = pd.DataFrame()
    if st.button("Cluster News Events"):
        # TODO: dynamically compute k
        st.session_state.data = news_connector.Cluster_Articles(k=int(nr_clusters), data=st.session_state.data.copy())
    
    # display clusters
    if 'cluster' in st.session_state.data:
        # display dataframe header
        st.write("First 20 entries in data with clusters:")
        st.dataframe(data=st.session_state.data[['title', 'cluster']].head(5))
    
        clustered_path = st.text_area("Path", value="clustered_articles1.csv")
        if st.button("Save Clusters"):
            st.session_state.data.to_csv(os.path.join("data/", str(clustered_path)))
    
    
    
######### 4. Article Statistics #########
line4_title_spacer1, line4_title, line4_title_spacer2 = st.columns((0.1, 3.2, 0.1))

line4_spacer1, line4_1, line4_space, line4_2, line4_spacer2 = st.columns((0.1, 1.5, 0.2, 1.5, 0.1))

with line4_title:
    st.header("Analyze Articles")
    if 'cluster' not in st.session_state.data:
        st.warning("Data has not yet been clustered.")
        st.stop()
    
    # event to display information about
    event_selected = st.number_input("Event Selected", 0, nr_clusters, value=0)


with line4_1:
    st.subheader("Who's Reporting?")
    st.markdown("Comparison of what news agencies are reporting on this event")
    
    counts = st.session_state.data['publication'].value_counts()
    counts_df = pd.DataFrame({'publication':counts.index, 'count':counts.values})
    #reportings_count_df = pd.DataFrame([[st.session_state.data['name'].unique()],[]])
    #reportings_df = pd.DataFrame(st.session_state.data.mask(st.session_state.data['cluster'] == event_selected).dropna().value_counts()).reset_index()
    #reportings_df.columns = ["Agency", "Count"]
    #reportings_df = reportings_df.sort_values(by="Year")
    fig = px.bar(
        counts_df,
        x="publication",
        y="count",
        title="Published articles on event",
        color_discrete_sequence=["#9EE6CF"],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    st.subheader("Reporting Sentiment")
    # Kaggle
    st.markdown("..")

with line4_2:
    st.subheader("Summary of Reporting")
    # GPT
    st.markdown("..")
    
    st.subheader("Differences in Reporting")
    st.markdown("..")