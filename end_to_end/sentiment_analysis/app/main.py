from transformers import pipeline
from fastapi import FastAPI

nlp = pipeline(task='sentiment-analysis',
               model='nlptown/bert-base-multilingual-uncased-sentiment')

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}


@app.get('/predict/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)


def analyze_sentiment(text):
    """Get and process result"""

    result = nlp(text)
    
    sent = ''
    
    if (result[0]['label'] == '1 star'):
        sent = 'Very Negative'
    elif (result[0]['label'] == '2 star'):
        sent = 'Negative'
    elif (result[0]['label'] == '3 stars'):
        sent = 'Neutral'
    elif (result[0]['label'] == '4 stars'):
        sent = 'Positive'
    else:
        sent = 'Very Positive'
        
    prob = result[0]['score']
    
    return {'sentiment': sent, 'probability': prob}