import io
import logging
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from config import DEFAULT_VOICE
import torchaudio as ta
import torch
import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)
router = APIRouter()

# Download required NLTK data packages at module import time
# 'punkt' for basic sentence tokenization, 'punkt_tab' for improved accuracy in newer versions
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@router.post("/tts/")
async def tts_long(
    request: Request,
    text: str = Form(..., description="Long text input for speech synthesis (stories, articles, etc.)")
):
    """
    Endpoint for generating speech from long text.
    Uses smart chunking to improve performance and adds natural pauses between chunks.
    Only accepts text input — voice is fixed to the configured default.
    """
    # Retrieve the globally loaded Chatterbox model from app state
    model = request.app.state.model or getattr(request.app, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model is still loading. Please try again in a few seconds."
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    # Fixed voice path — no user override for security and consistency
    voice_path = DEFAULT_VOICE

    try:
        # Step 1: Tokenize text into sentences for intelligent splitting
        sentences = sent_tokenize(text.strip())

        # Step 2: Create chunks with a safe character limit to avoid model context overflow
        chunks = []
        current_chunk = ""
        max_chars = 300  # Optimal balance for performance and quality

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Long TTS request processed: split into {len(chunks)} chunks")

        # Step 3: Generate audio for each chunk
        audio_chunks = []
        sample_rate = model.sr  # Model's native sample rate (typically 24000 Hz)

        for chunk in chunks:
            wav = model.generate(text=chunk, audio_prompt_path=voice_path)
            wav = wav.cpu().squeeze(0)  # Remove batch dimension: (seq,) shape
            audio_chunks.append(wav)

        # Step 4: Add natural silence between chunks for better prosody
        silence_duration = 0.4  # seconds — adjust for shorter/longer pauses if needed
        silence_samples = int(sample_rate * silence_duration)
        silence = torch.zeros(silence_samples)

        full_audio_parts = []
        for i, audio in enumerate(audio_chunks):
            full_audio_parts.append(audio)
            if i < len(audio_chunks) - 1:  # No silence after the last chunk
                full_audio_parts.append(silence)

        # Concatenate all parts into final waveform
        full_wav = torch.cat(full_audio_parts).unsqueeze(0)  # Add batch dimension for saving

        # Step 5: Save to in-memory buffer and return as streaming response
        buffer = io.BytesIO()
        ta.save(buffer, full_wav, sample_rate, format="wav")
        buffer.seek(0)

        headers = {
            "Content-Disposition": 'attachment; filename="speech.wav"'
        }

        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    except Exception as e:
        logger.error(f"Long TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")