import os
from dotenv import load_dotenv
import torch
import logging
from pathlib import Path

# Load environment variables from .env file (if present)
# This allows configuration via environment variables or .env for flexibility in production
load_dotenv()

# Configure logging - standard format for console output in development and production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Detect and set computation device (GPU if available, otherwise CPU)
# Utilizes PyTorch's CUDA detection for optimal performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Computation device selected: {DEVICE}")

# Retrieve default voice file path from environment variable
# Falls back to a hardcoded path if DEFAULT_VOICE is not set
# Strips any surrounding quotes and whitespace for robustness
DEFAULT_VOICE = os.getenv(
    "DEFAULT_VOICE",
    r"C:\Users\neovotech\EN_TTS_CHATTER_BOX\ElevenLabs_Text_to_Speech_audio (2).mp3"
).strip().strip('"').strip("'")

# Convert to Path object for better path handling (optional but recommended)
DEFAULT_VOICE_PATH = Path(DEFAULT_VOICE)

# Validate the existence of the default voice file at startup
# Logs a warning if missing (critical for TTS functionality)
if not DEFAULT_VOICE_PATH.is_file():
    logger.warning(
        f"Default voice file not found at configured path: {DEFAULT_VOICE_PATH.resolve()}"
    )
else:
    logger.info(
        f"Default voice file successfully located: {DEFAULT_VOICE_PATH.resolve()}"
    )