import os
import urllib.request

import numpy as np
import pandas as pd
#import plotly.express as px
import requests

import streamlit as st
#import xmltodict
#from mitosheet.streamlit.v1 import spreadsheet
#from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
#from streamlit_lottie import st_lottie

st.set_page_config(page_title="News Comparison App", layout="wide")

###
# 1. Scrape Data
# 2. Select Datasets (all-the-news / scraped / ...)
# 3. Process Similar Articles
# 4. Article Statistics
###

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

with line1_1:
    st.header("Optional: Scrape Data")#**{}**".format(user_name))
    st.markdown("TODO: implement this in app")
    
    
######### 2. Select Datasets #########
line2_spacer1, line2_1, line2_spacer2 = st.columns((0.1, 3.2, 0.1))
selected_data = []
data_frames = []
data = []

with line2_1:
    st.header("Select Datasets")
    
    # select box for datasets
    datasets = [file for file in os.listdir("./data/") if file.endswith(('csv'))]
    if len(datasets) == 0:
        st.write("No datasets found in `data` folder.")
        st.stop()
    selected_data = st.multiselect("Select", datasets)

    # load data button
    if st.button("Load data"):
        for data_path in selected_data:
            data_frames.append(pd.read_csv(os.path.join('./data/', data_path)))
        data = pd.concat(data_frames)
        data['date'] = pd.to_datetime(data['date'])
        data = data.drop(['Unnamed: 0', 'year', 'month', 'url'], axis=1)
    
    if len(data) > 0:
        # display dataframe header
        st.dataframe(data=data)
        
        # display data statistics
        #..
    
    
######### 3. Process Similar Articles (News Connector) #########
line3_spacer1, line3_1, line3_spacer2 = st.columns((0.1, 3.2, 0.1))

with line3_1:
    if len(data) == 0:
        st.write("No datasets loaded.")
        st.stop()

    st.header("Process Similar Articles")
    
    
    
######### 4. Article Statistics #########
line4_spacer1, line4_1, line4_spacer2 = st.columns((0.1, 3.2, 0.1))

with line4_1:
    if len(data) == 0:
        st.write("No datasets loaded.")
        st.stop()

    st.header("Analyze Articles")
