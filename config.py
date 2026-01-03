import os
from dotenv import load_dotenv
import torch
import logging
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Automatic device selection: GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Selected computation device: {DEVICE}")

# Default voice path from environment (secure and configurable)
DEFAULT_VOICE = os.getenv(
    "DEFAULT_VOICE",
    r"C:\Users\neovotech\Chatter_EN\voice\ElevenLabs_Text_to_Speech_audio (2).mp3"
).strip().strip('"').strip("'")

# Use Path for reliable file handling
DEFAULT_VOICE_PATH = Path(DEFAULT_VOICE)

# Validate voice file existence at startup
if not DEFAULT_VOICE_PATH.is_file():
    logger.error(f"CRITICAL: Default voice file not found: {DEFAULT_VOICE_PATH.resolve()}")
    raise FileNotFoundError(f"Voice file missing: {DEFAULT_VOICE_PATH}")
else:
    logger.info(f"Default voice file loaded: {DEFAULT_VOICE_PATH.resolve()}")