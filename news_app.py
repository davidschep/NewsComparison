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
import news_webscraping
import news_dataloader
import news_summarization
import news_sentimentanalysis

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



###########################################################################################
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
        news_webscraping.scraper(filename="new_scraping_data.csv", publication_list=select_agencies, max_limit_num_articles=int(slider_articles_nr))   
    
    
    
###########################################################################################
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
    st.session_state.selected_data = st.selectbox("Select", st.session_state.datasets)

    # load data button
    if st.button("Load data"):
        DATA_LIMIT = 520 # how many items to load
        st.session_state.data = news_dataloader.load_data(str(st.session_state.selected_data), DATA_LIMIT)
    
    if len(st.session_state.data) > 0:
        # display dataframe header
        st.write("First 20 entries in data (size " + str(len(st.session_state.data)) + "):")
        st.dataframe(data=st.session_state.data.head(20))
        
        # display data statistics
        #..
    
    
    
###########################################################################################
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
        st.write("First 5 entries in data with clusters:")
        st.dataframe(data=st.session_state.data[['title', 'cluster']].head(5))
        #indices = st.session_state.data['cluster'] == "0"
        #st.dataframe(data=st.session_state.data.loc[indices][['title', 'cluster', 'content']])
        #st.dataframe(data=st.session_state.data[['title', 'cluster', 'content']].loc[st.session_state.data['cluster'] == 0].head(10))

        clustered_path = st.text_area("Path", value="clustered_articles1.csv")
        if st.button("Save Clusters"):
            news_dataloader.save_data(str(clustered_data), st.session_state.data)
    
    
    
###########################################################################################
######### 4. Article Statistics #########
line4_title_spacer1, line4_title, line4_title_spacer2 = st.columns((0.1, 3.2, 0.1))

line4_spacer1, line4_1, line4_space, line4_2, line4_spacer2 = st.columns((0.1, 1.5, 0.2, 1.5, 0.1))

with line4_title:
    st.header("Analyze Articles")
    if 'cluster' not in st.session_state.data:
        st.warning("Data has not yet been clustered.")
        st.stop()
    
    # event to display information about (if events are filled out)
    if 'event' in st.session_state.data:
        event_selected = st.number_input("Event Selected", 1, nr_clusters, value=1)
        reportings_indices = st.session_state.data['event'] == event_selected
    # else show clusters
    elif 'cluster' in st.session_state.data:
        event_selected = st.number_input("Event Selected", 0, nr_clusters, value=0)
        reportings_indices = st.session_state.data['cluster'] == event_selected
        
    # TODO: change from number_input into select box with names
    # TODO: list one article title as description
    
    # list all article titles
    publication = 'publication' if 'publication' in st.session_state.data else 'name'
    st.dataframe(data=st.session_state.data.loc[reportings_indices][[publication, 'title']])

with line4_1:
    st.subheader("Who's Reporting?")
    st.markdown("_Comparison of what news agencies are reporting on this event:_")
    
    counts = st.session_state.data.loc[reportings_indices][publication].value_counts()
    counts_df = pd.DataFrame({publication:counts.index, 'count':counts.values})
    #reportings_count_df = pd.DataFrame([[st.session_state.data['name'].unique()],[]])
    #reportings_df.columns = ["Agency", "Count"]
    #reportings_df = reportings_df.sort_values(by="Year")
    fig = px.bar(
        counts_df,
        x=publication,
        y="count",
        title="Published articles on event",
        color_discrete_sequence=["#9EE6CF"],
        labels={publication:'News Agency', 'count':'Number of Articles'}
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    st.subheader("Reporting Sentiment")
    titles = []
    sentiments = []
    for index, row in st.session_state.data.loc[reportings_indices].iterrows():
        title = row[publication] + ": " + row['title']
        titles.append(title)
        score = news_sentimentanalysis.get_sentiment_scores(row['content'])['compound']
        sentiments.append(score)
        color = "green" if score>0 else "red"
        st.write(row[publication] + ": :" + color + "[" + row['title'] + "] has sentiment score " + str(score))
    sentiment_df = pd.DataFrame({'titles':titles, 'sentiments':sentiments})
    

with line4_2:
    st.subheader("Summary of Reporting")
    st.markdown("_A summary comparison of how different news agencies report on this event:_")
    if 'summary' not in st.session_state.data:
        st.session_state.data["summary"] = "-"
    save_df = False
    for index, row in st.session_state.data.loc[reportings_indices].iterrows():
        if row['summary'] == "-":
            save_df = True
            st.session_state.data.iloc[index, st.session_state.data.columns.get_loc('summary')] = news_summarization.get_summary(row['content'])
        st.markdown("*" + row[publication] + "*: " + st.session_state.data.iloc[index, st.session_state.data.columns.get_loc('summary')])
    if save_df:
        print("Updating csv with summaries: ", str(st.session_state.selected_data))
        news_dataloader.save_data(str(st.session_state.selected_data), st.session_state.data)

    
    #st.subheader("Differences in Reporting")
    #st.markdown("..")