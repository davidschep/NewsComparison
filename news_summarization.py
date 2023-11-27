#!/usr/bin/env python
# coding: utf-8

import requests
import time

def query(input_text, retries=5, delay=5):
    # using FalconAI's text summarization model
    API_URL = 'https://api-inference.huggingface.co/models/Falconsai/text_summarization'
    headers = {'Authorization': 'Bearer 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'}
    
    for i in range(retries):
        input_ = {"inputs":input_text,}
        response = requests.post(API_URL, headers=headers, json=input_)
        if response.status_code == 200:
            return response.json()
        else:
            # model might take some time to initialize and load
            print(f"Retry {i+1}/{retries}...")
            print("Model is loading, waiting for", delay, "seconds")
            time.sleep(delay)
    return None

def get_summary(input_text):
    #Final summary function - takes text as input and returns summary
    output_summary = query(input_text)   
    return output_summary[0]['summary_text']