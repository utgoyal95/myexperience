import streamlit
import requests
import json
import pandas as pd

       
def run():
    streamlit.title("Sentiment Prediction")
    text = streamlit.text_input('Please tell me how you feel')
    
    query = {'text':text}

    if streamlit.button("Analyze:"):
        response = requests.get('http://127.0.0.1:8000/predict/', params=query)
        sentiment = response.json()['sentiment']
        prob = response.json()['probability']*100
        
        streamlit.success(f"This is a very: {sentiment} sentiment, with a confidence score of {prob:.0f}%")
    
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()