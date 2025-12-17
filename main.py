import logging
import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI
from tts_long import router as tts_long_router  # Router for long-text TTS endpoint
from chatterbox.tts import ChatterboxTTS  # Chatterbox TTS model class
from config import DEVICE, DEFAULT_VOICE  # Device (GPU/CPU) and default voice path

# Logging configuration - standard format for production/console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI application instance
app = FastAPI(
    title="Chatterbox Long TTS API",
    version="1.0.0",
    description="Optimized TTS service for long texts (with chunking and natural pauses)"
)

# Global model variable - loaded once at server startup
model = None

@app.on_event("startup")
async def load_model():
    """Load the Chatterbox TTS model when the server starts"""
    global model
    try:
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        logger.info(f"✅ Chatterbox TTS model successfully loaded on {DEVICE}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

# Include the TTS router
app.include_router(tts_long_router)

# Store model in app state for access in other modules (optional fallback)
app.state.model = None

# Health check endpoint - used for monitoring and production readiness
start_time = time.time()

@app.get("/health")
async def health_check():
    """Health check endpoint returning server status and basic info"""
    global model
    uptime = time.time() - start_time
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat(),
        "default_voice": DEFAULT_VOICE
    }

# Run the server when executed directly
if __name__ == "__main__":
    logger.info("Starting Long TTS API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )