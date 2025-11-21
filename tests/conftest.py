import sys
import types
from unittest.mock import MagicMock



def mock_module(name):
    """
    Creates a mock module that passes python's importlib checks.
    """
    m = types.ModuleType(name)
    m.__spec__ = MagicMock()
    m.__spec__.name = name
    m.__spec__.loader = MagicMock()
    m.__path__ = []  
    return m


# Mock TA-Lib

mock_talib = mock_module("talib")
mock_talib.SMA = MagicMock(return_value=[100.0] * 100)
mock_talib.EMA = MagicMock(return_value=[100.0] * 100)
mock_talib.RSI = MagicMock(return_value=[50.0] * 100)
mock_talib.MACD = MagicMock(return_value=( [0.0]*100, [0.0]*100, [0.0]*100 ))
mock_talib.ATR = MagicMock(return_value=[10.0] * 100)
mock_talib.BBANDS = MagicMock(return_value=( [110.0]*100, [100.0]*100, [90.0]*100 ))
mock_talib.OBV = MagicMock(return_value=[1000.0] * 100)

for pattern in ['CDLENGULFING', 'CDLHARAMI', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLDOJI', 
                'CDLINVERTEDHAMMER', 'CDLSPINNINGTOP', 'CDLMARUBOZU']:
    setattr(mock_talib, pattern, MagicMock(return_value=[0] * 100))

sys.modules["talib"] = mock_talib


# Mock Datasets

mock_datasets = mock_module("datasets")
mock_datasets.load_dataset = MagicMock()
sys.modules["datasets"] = mock_datasets

# Mock NLTK

mock_nltk = mock_module("nltk")
mock_nltk.download = MagicMock()
mock_nltk.sent_tokenize = MagicMock(side_effect=lambda t: t.split(". "))

mock_nltk_tokenize = mock_module("nltk.tokenize")
mock_nltk_tokenize.sent_tokenize = mock_nltk.sent_tokenize

mock_nltk_data = mock_module("nltk.data")
mock_nltk_data.find = MagicMock(return_value="fake/path")

sys.modules["nltk"] = mock_nltk
sys.modules["nltk.tokenize"] = mock_nltk_tokenize
sys.modules["nltk.data"] = mock_nltk_data

# Mock Qdrant 

mock_qdrant = mock_module("qdrant_client")
mock_qdrant.QdrantClient = MagicMock()

mock_qdrant_models = mock_module("qdrant_client.models")
mock_qdrant_models.Filter = MagicMock()
mock_qdrant_models.FieldCondition = MagicMock()
mock_qdrant_models.MatchText = MagicMock()

mock_qdrant_http = mock_module("qdrant_client.http")

mock_qdrant_http_models = mock_module("qdrant_client.http.models")
mock_qdrant_http_models.Distance = MagicMock()
mock_qdrant_http_models.VectorParams = MagicMock()
mock_qdrant_http_models.PointStruct = MagicMock()

sys.modules["qdrant_client"] = mock_qdrant
sys.modules["qdrant_client.models"] = mock_qdrant_models
sys.modules["qdrant_client.http"] = mock_qdrant_http
sys.modules["qdrant_client.http.models"] = mock_qdrant_http_models