# Crypto AI Analytics Platform

A production-grade cryptocurrency analytics system that leverages LLM orchestration and specialized AI models to deliver actionable insights from multi-modal data sources.

---

## Overview

The Crypto AI Analytics Platform combines advanced machine learning, natural language processing, and blockchain analytics into a unified system. Built on a monolithic architecture with Model Context Protocol (MCP) orchestration, the platform processes market data, social sentiment, and on-chain metrics to generate interpretable, citation-backed insights.

---

## Core Capabilities

### Intelligent Query Processing
- Natural language interface powered by multi-LLM orchestration (Gemini, Groq, OpenRouter)
- Dynamic orchestration of specialized analytical modules via MCP
- Hybrid query classification with context-aware response generation
- Source attribution and citation trails

### Time-Series Forecasting
- Prophet-based price prediction with trend decomposition
- CNN-LSTM deep learning model for sequential pattern recognition
- Temporal Fusion Transformer (TFT) for multi-horizon forecasting
- SARIMAX for statistical time-series modeling
- SHAP explainability for model transparency

### Sentiment Analysis
- Fine-tuned DistilRoBERTa on cryptocurrency-specific corpus
- VADER sentiment scoring for news and social content
- Retrieval-Augmented Generation (RAG) for grounded insights
- Vector-based semantic search with Qdrant

### On-Chain Analytics
- Real-time whale transaction monitoring via Infura
- Exchange flow analysis (deposits vs. withdrawals)
- Technical analysis pattern detection (RSI, MACD, etc.)
- Network activity metrics from Ethereum

### Multi-Source Data Integration
- Market data aggregation via CCXT (100+ exchanges)
- Social sentiment from Reddit (PRAW) and CryptoPanic
- Fear & Greed Index from Alternative.me
- Real-time trade polling and OHLCV backfill

---

## Architecture

### System Design

The platform employs a three-tier architecture optimized for low-latency inter-module communication:

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface Layer                      │
│                                                             │
│   ┌─────────────────┐    ┌─────────────────────────────┐    │
│   │  React Frontend │    │      REST API (FastAPI)     │    │
│   │  Vite + Tailwind│    │                             │    │
│   └─────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Core Processing Layer                      │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              LLM Agent Orchestrator                 │   │
│   │         (Gemini / Groq / OpenRouter)                │   │
│   │              Model Context Protocol                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                              │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│   │  Forecasting  │  │   Sentiment   │  │   On-Chain    │   │
│   │    Module     │  │    Module     │  │    Module     │   │
│   │               │  │               │  │               │   │
│   │ Prophet       │  │ DistilRoBERTa │  │ Whale Alerts  │   │
│   │ CNN-LSTM      │  │ RAG Pipeline  │  │ Exchange Flows│   │
│   │ TFT           │  │ VADER         │  │ TA Patterns   │   │
│   │ SARIMAX       │  │               │  │               │   │
│   └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Data & Integration Layer                    │
│                                                             │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
│   │TimescaleDB│  │   Redis   │  │  Qdrant   │  │ MLflow  │  │
│   │  (OHLCV)  │  │  (Cache)  │  │ (Vectors) │  │(Models) │  │
│   └───────────┘  └───────────┘  └───────────┘  └─────────┘  │
│                                                             │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
│   │   CCXT    │  │  Infura   │  │  Reddit   │  │CryptoPanic │
│   │(Exchanges)│  │(Ethereum) │  │  (PRAW)   │  │ (News)  │  │
│   └───────────┘  └───────────┘  └───────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────┘
```


### Agent Workflow

1. **Query Reception**: User submits natural language query through React frontend or API
2. **Intent Classification**: Hybrid classifier analyzes query and determines required analytical modules
3. **Tool Orchestration**: Agent invokes relevant MCP tools in parallel (forecasting, sentiment, on-chain)
4. **Data Processing**: Each module executes specialized analysis on requested data
5. **Result Synthesis**: Multi-LLM synthesis aggregates outputs and generates explainable response
6. **Response Delivery**: Final answer with citations, visualizations, and trading signals returned to user

### Design Rationale

The monolithic architecture was chosen to optimize for:

| Aspect | Benefit |
|--------|---------|
| **Performance** | Internal function calls eliminate network overhead between modules |
| **Maintainability** | Unified deployment, logging, and debugging infrastructure |
| **Data Sharing** | Modules share database connections and memory without serialization costs |
| **Modularity** | MCP protocol provides clear service boundaries without distributed system complexity |

---

## Key Features

### Explainable AI
- SHAP values for forecast feature importance
- LIME explanations for model interpretability
- RAG citations linking to source documents
- Transparent model decision-making process

### Real-Time Processing
- Async data ingestion with Celery task queue
- Redis caching for low-latency query processing (sub-second for cached queries)
- Streaming responses for long-running analyses

### Data Validation
- Strict schema enforcement with Pydantic models
- Input sanitization and type checking
- Comprehensive error handling with detailed logging

### Extensibility
- Modular design allows easy addition of new analytical tools
- MCP protocol standardizes tool interfaces
- Support for custom models and data sources
- MLflow model registry for versioning and deployment

---

## Technology Stack

### Backend
| Category | Technologies |
|----------|--------------|
| **Framework** | FastAPI, Python 3.10+ |
| **LLM Orchestration** | Gemini, Groq, OpenRouter, Model Context Protocol |
| **Forecasting** | Prophet, TensorFlow/Keras (CNN-LSTM), PyTorch Forecasting (TFT), Statsmodels (SARIMAX) |
| **NLP** | DistilRoBERTa (Transformers), Sentence-Transformers, VADER Sentiment |
| **Vector Store** | Qdrant |
| **Explainability** | SHAP, LIME |
| **Validation** | Pydantic |
| **MLOps** | MLflow |

### Frontend
| Category | Technologies |
|----------|--------------|
| **Framework** | React 18 |
| **Build Tool** | Vite |
| **Styling** | Tailwind CSS |
| **Charts** | Recharts |
| **Icons** | Lucide React |

### Infrastructure
| Category | Technologies |
|----------|--------------|
| **Time-Series DB** | TimescaleDB (PostgreSQL) |
| **Cache** | Redis |
| **Vector DB** | Qdrant |
| **Model Registry** | MLflow |
| **Blockchain** | Web3.py, Infura |
| **Containerization** | Docker, Docker Compose |

### Data Sources
| Category | Technologies |
|----------|--------------|
| **Market Data** | CCXT  |
| **Social Media** | Reddit (PRAW) |
| **News** | CryptoPanic API |
| **Sentiment Index** | Alternative.me (Fear & Greed) |
| **On-Chain** | Infura (Ethereum) |

---

## Project Structure

```
crypto-analytics-platform/
├── src/
│   ├── core/                 # Configuration, database, logging
│   ├── data/                 # Data ingestion and storage
│   │   ├── ingestion/        # CCXT, Reddit, CryptoPanic, Ethereum clients
│   │   └── storage/          # CRUD operations and models
│   ├── modules/
│   │   ├── agent/            # LLM orchestrator and query classifier
│   │   ├── forecasting/      # Prophet, CNN-LSTM, TFT, SARIMAX models
│   │   ├── sentiment/        # DistilRoBERTa, RAG pipeline
│   │   ├── onchain/          # Whale alerts, exchange flows, TA patterns
│   │   └── dashboard/        # Frontend data serializers
│   ├── mcp_servers/          # MCP tool servers
│   ├── services/             # FastAPI routers
│   └── utils/                # Cache, GCS loader
├── frontend/                 # React + Vite application
├── infrastructure/           # Dockerfiles, nginx, SQL scripts
├── tests/                    # Test suite
└── notebook/                 # Jupyter notebooks
```

---

## Use Cases

### Institutional Applications
- Trading desk decision support with explainable forecasts
- Risk assessment using multi-modal sentiment indicators
- Compliance-ready analytics with full citation trails

### Research Applications
- Reproducible analysis with documented data sources
- Model comparison and benchmarking
- Behavioral pattern analysis across market conditions

### Individual Traders
- Natural language queries for complex market questions
- Automated whale activity alerts
- Sentiment-aware price predictions
- Technical analysis pattern detection

---

## Future Development

### Phase 1: Enhanced Forecasting
- Ensemble modeling combining multiple algorithms
- Automated hyperparameter optimization with Optuna
- Real-time model retraining pipeline

### Phase 2: Advanced Analytics
- Wallet clustering for behavioral segmentation
- Chart pattern recognition using computer vision
- Smart contract interaction analysis

### Phase 3: Infrastructure Evolution
- Selective microservice extraction for high-load modules
- Multi-region deployment with edge caching
- Enhanced monitoring and observability

---

## Research Foundation

This platform is built on research investigating the performance trade-offs between monolithic and microservice architectures in AI-driven financial systems. The implementation validates that monolithic designs with internal modularity can achieve superior latency and maintainability characteristics for certain use cases.

**Key findings include:**
- 50x reduction in inter-module communication latency
- 38% improvement in complex query orchestration time
- 28 percentage point increase in output faithfulness through RAG

For detailed methodology and experimental results, refer to the accompanying research paper:  
[*"Crypto AI Analytics Platform: A Monolithic LLM-Orchestrated Architecture for Multi-Modal Cryptocurrency Analysis"*](docs/CryptoAI_Research_Paper.pdf)

---

## About

Developed by **Muhammed Sarfras P C** as a demonstration of advanced AI system architecture, LLM orchestration, and explainable machine learning in financial applications.

---

## Disclaimer

> ⚠️ This platform is designed for educational and research purposes. The analytics and predictions provided should not be construed as financial advice. Cryptocurrency investments carry significant risk.

---

