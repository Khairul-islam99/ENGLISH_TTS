import logging
import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI

from tts_long import router as tts_long_router
from chatterbox.tts import ChatterboxTTS
from config import DEVICE

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="Chatterbox Long-Form TTS API",
    description="High-quality text-to-speech service optimized for long content with natural flow.",
    version="1.0.0"
)

# Global model reference
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on server startup"""
    global model
    try:
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        app.state.model = model  # Make available to routes
        logger.info(f"TTS model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.critical(f"Failed to load TTS model: {e}")
        raise

# Include API routes
app.include_router(tts_long_router)

# Server uptime tracking
start_time = time.time()

@app.get("/health")
async def health_check():
    """Health endpoint for monitoring and load balancers"""
    uptime = time.time() - start_time
    return {
        "status": "healthy" if model is not None else "initializing",
        "model_loaded": model is not None,
        "device": DEVICE,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Development server entry point
if __name__ == "__main__":
    logger.info("Starting Chatterbox Long TTS API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )