#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')

def get_sentiment_scores(text):
    """
    Returns a dictionary like {'neg': 0.073, 'neu': 0.799, 'pos': 0.127, 'compound': 0.9935} where keys correspond to the negative, neutral, positive, and compound scores
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)
