import logging
import os
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL_ID = "sarfras/crypto-sentiment-distilroberta"


class SentimentClassifier:
    def __init__(self, model_path: str = None):
        """
        model_path can be either:
          - a local directory with a saved HF model, or
          - a Hugging Face repo ID (e.g. 'sarfras/crypto-sentiment-distilroberta').

        If None, we resolve in this order:
          1) local './saved/finetuned_model' if it exists
          2) env var SENTIMENT_MODEL_ID
          3) default HF repo DEFAULT_SENTIMENT_MODEL_ID
        """
        if model_path is None:
            local_dir = os.path.join(os.path.dirname(__file__), "saved", "finetuned_model")
            if os.path.isdir(local_dir):
                model_path = local_dir
            else:
                model_path = os.getenv("SENTIMENT_MODEL_ID", DEFAULT_SENTIMENT_MODEL_ID)

        self.model_path = model_path
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                return_all_scores=True,
                device=-1,  
            )
            logger.info(f"Sentiment model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str):
        results = self.classifier(text)
        return self._format_prediction(results[0])
    
    def predict_batch(self, texts: list):
        results = self.classifier(texts)
        return [self._format_prediction(result) for result in results]
    
    def _format_prediction(self, result):
        formatted = {}
        
        for score in result:
            label = score['label']
            
            if label.startswith('LABEL_'):
                label_id = int(label.split('_')[-1])
                if label_id == 0:
                    label_name = 'BEARISH'
                elif label_id == 1:
                    label_name = 'BULLISH'
                else:
                    label_name = 'NEUTRAL'
            else:
                label_name = label
            
            formatted[label_name] = float(score['score'])
        
        for sentiment in ['BEARISH', 'BULLISH', 'NEUTRAL']:
            if sentiment not in formatted:
                formatted[sentiment] = 0.0
        
        top_label = max(formatted.items(), key=lambda x: x[1])
        formatted['top_sentiment'] = top_label[0]
        formatted['top_confidence'] = top_label[1]
        
        return formatted
    
    def quick_predict(self, text: str):
        result = self.predict(text)
        return {
            'sentiment': result['top_sentiment'],
            'confidence': result['top_confidence'],
            'bearish_score': result.get('BEARISH', 0),
            'bullish_score': result.get('BULLISH', 0),
            'neutral_score': result.get('NEUTRAL', 0),
        }

_sentiment_classifier = None

def get_sentiment_classifier(model_path: str = None):
    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = SentimentClassifier(model_path)
    return _sentiment_classifier

def analyze_sentiment(text: str, model_path: str = None):
    classifier = get_sentiment_classifier(model_path)
    return classifier.quick_predict(text)

def analyze_sentiment_batch(texts: list, model_path: str = None):
    classifier = get_sentiment_classifier(model_path)
    return classifier.predict_batch(texts)

if __name__ == "__main__":
    classifier = get_sentiment_classifier()
    
    test_texts = [
        "Bitcoin is going to the moon! ",
        "Market crash incoming, sell everything!",
        "The crypto market shows mixed signals today.",
        "ETH breaking resistance levels, bullish momentum building.",
        "Fear and uncertainty dominating the market sentiment."
    ]
    
    print("Testing Sentiment Classifier:")
    print("=" * 50)
    
    for text in test_texts:
        try:
            result = classifier.quick_predict(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print(f"Scores - Bearish: {result['bearish_score']:.3f}, "
                  f"Bullish: {result['bullish_score']:.3f}, "
                  f"Neutral: {result['neutral_score']:.3f}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing text: {e}")
