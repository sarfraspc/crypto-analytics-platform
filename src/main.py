from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from core.config import settings
from core.logging_config import setup_logging
from core.exceptions import APIError, DatabaseError, ModelError
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto AI Analytics Platform",
    version="0.1.0",
    description="MCP-integrated monolithic crypto analytics platform"
)

# Exception handlers
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    logger.error(f"APIError: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code or 500,
        content={"error": exc.message},
    )

@app.exception_handler(DatabaseError)
async def db_error_handler(request: Request, exc: DatabaseError):
    logger.error("DatabaseError occurred")
    return JSONResponse(status_code=500, content={"error": "Database error"})

@app.exception_handler(ModelError)
async def model_error_handler(request: Request, exc: ModelError):
    logger.error("ModelError occurred")
    return JSONResponse(status_code=500, content={"error": "Model error"})

# Health check
@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok", "env": settings.APP_ENV}
