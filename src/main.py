from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
import uvicorn
import logging
from datetime import datetime

from services.agent import router as agent_router
from services.price import router as price_router
from services.sentiment import router as sentiment_router
from services.onchain import router as onchain_router
from services.dashboard import router as dashboard_router

from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Analytics API", version="1.0.0")

# CORS with env flexibility
cors_origins = settings.ALLOWED_ORIGINS.split(",") if hasattr(settings, 'ALLOWED_ORIGINS') else ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(price_router)
app.include_router(sentiment_router)
app.include_router(onchain_router)
app.include_router(dashboard_router)

@app.get("/")
async def root():
    return {"message": "Crypto Analytics API is running", "env": settings.APP_ENV, "timestamp": datetime.now().isoformat()}

@app.get("/healthz")
async def healthz():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    logger.info(f"Starting FastAPI on port {settings.APP_PORT} in {settings.APP_ENV} mode")
    uvicorn.run(app, host="0.0.0.0", port=settings.APP_PORT)