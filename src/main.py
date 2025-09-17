from fastapi import FastAPI
from src.core.config import settings

app = FastAPI(
    title="Crypto AI Analytics Platform",
    version="0.1.0",
    description="MCP-integrated monolithic crypto analytics platform"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "env": settings.APP_ENV} 