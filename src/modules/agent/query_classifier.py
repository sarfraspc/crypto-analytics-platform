from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryClassification:
    query_type: str
    requires_forecast: bool = False
    requires_sentiment: bool = False
    requires_rag: bool = False
    requires_onchain: bool = False
    
    def dict(self):
        return self.__dict__

class QueryClassifier:
    # ordered on priority: long_context > reasoning > real_time
    LONG_CONTEXT = ['report', 'summary', 'overview', 'history', 'backtest', 'strategy', 'performance', 'detailed', 'comprehensive']
    REASONING = ['why', 'explain', 'reason', 'cause', 'because', 'analysis', 'drop', 'rise', 'will', 'predict', 'should']
    REAL_TIME = ['price', 'current', 'now', 'latest', 'today', 'live', 'what is', 'how much', 'score']
    
    def classify_query(self, question: str) -> QueryClassification:
        q = question.lower()
        
        if any(k in q for k in self.LONG_CONTEXT):
            qtype = "long_context"
        elif any(k in q for k in self.REASONING):
            qtype = "reasoning"
        elif any(k in q for k in self.REAL_TIME):
            qtype = "real_time"
        else:
            qtype = "reasoning"  # default to reasoning
        
        req_forecast = any(k in q for k in ['forecast', 'predict', 'future', 'next', 'horizon'])
        req_sentiment = any(k in q for k in ['sentiment', 'bullish', 'bearish', 'mood', 'emotion'])
        req_rag = any(k in q for k in ['why', 'news', 'article', 'event', 'context'])
        req_onchain = any(k in q for k in ['whale', 'on-chain', 'flow', 'transfer', 'exchange'])
        
        # Force all sources for reasoning if nothing specific
        if qtype == "reasoning" and not any([req_forecast, req_sentiment, req_rag, req_onchain]):
            req_rag = req_sentiment = req_onchain = True
            
        return QueryClassification(
            query_type=qtype,
            requires_forecast=req_forecast,
            requires_sentiment=req_sentiment,
            requires_rag=req_rag,
            requires_onchain=req_onchain
        )