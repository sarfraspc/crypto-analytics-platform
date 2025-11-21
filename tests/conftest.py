import sys
import types
from unittest.mock import MagicMock


# GLOBAL MOCKS FOR CI/CD

""" 
These mocks run before ANY test files are imported.
This prevents "ModuleNotFoundError" in CI environments 
where heavy libraries (TA-Lib, Torch) aren't installed.
"""

# Mock TA-Lib 
mock_talib = types.ModuleType("talib")
mock_talib.SMA = MagicMock(return_value=[100.0] * 100)
mock_talib.EMA = MagicMock(return_value=[100.0] * 100)
mock_talib.RSI = MagicMock(return_value=[50.0] * 100)
mock_talib.MACD = MagicMock(return_value=( [0.0]*100, [0.0]*100, [0.0]*100 )) 
mock_talib.ATR = MagicMock(return_value=[10.0] * 100)
mock_talib.BBANDS = MagicMock(return_value=( [110.0]*100, [100.0]*100, [90.0]*100 ))
mock_talib.OBV = MagicMock(return_value=[1000.0] * 100)

# Mock candlestick patterns (return 0 for no pattern, 100 for bullish)
for pattern in ['CDLENGULFING', 'CDLHARAMI', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLDOJI', 
                'CDLINVERTEDHAMMER', 'CDLSPINNINGTOP', 'CDLMARUBOZU']:
    setattr(mock_talib, pattern, MagicMock(return_value=[0] * 100))

sys.modules["talib"] = mock_talib

# Mock Datasets 
mock_datasets = types.ModuleType("datasets")
mock_datasets.load_dataset = MagicMock()
sys.modules["datasets"] = mock_datasets

# Mock Qdrant 
mock_qdrant = types.ModuleType("qdrant_client")
mock_qdrant.QdrantClient = MagicMock()
mock_qdrant_models = types.ModuleType("qdrant_client.models")
mock_qdrant_models.Filter = MagicMock()
mock_qdrant_models.FieldCondition = MagicMock()
mock_qdrant_models.MatchText = MagicMock()
sys.modules["qdrant_client"] = mock_qdrant
sys.modules["qdrant_client.models"] = mock_qdrant_models
sys.modules["qdrant_client.http"] = MagicMock()
sys.modules["qdrant_client.http.models"] = MagicMock()

# Mock NLTK 
mock_nltk = types.ModuleType("nltk")
mock_nltk.download = MagicMock()
mock_nltk.sent_tokenize = MagicMock(side_effect=lambda text: text.split(". "))
sys.modules["nltk"] = mock_nltk