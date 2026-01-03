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

# Ensure required NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@router.post("/tts/")
async def generate_speech(
    request: Request,
    text: str = Form(..., description="Long text to convert to speech (stories, reports, books, etc.)")
):
    """
    Generate high-quality speech from long text using chunking for performance
    and natural silence pauses for realistic delivery.
    """
    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable — model is still initializing."
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    voice_path = DEFAULT_VOICE

    try:
        # Intelligent sentence splitting
        sentences = sent_tokenize(text.strip())

        # Chunking strategy
        chunks = []
        current_chunk = ""
        max_chars_per_chunk = 300

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < max_chars_per_chunk:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Processing long TTS request — {len(chunks)} chunks generated")

        # Generate audio per chunk
        audio_chunks = []
        sample_rate = model.sr

        for chunk in chunks:
            wav = model.generate(text=chunk, audio_prompt_path=voice_path)
            wav = wav.cpu().squeeze(0)
            audio_chunks.append(wav)

        # Insert natural pauses
        pause_duration = 0.4  # seconds
        pause_samples = int(sample_rate * pause_duration)
        pause = torch.zeros(pause_samples)

        full_audio = []
        for i, chunk_audio in enumerate(audio_chunks):
            full_audio.append(chunk_audio)
            if i < len(audio_chunks) - 1:
                full_audio.append(pause)

        final_waveform = torch.cat(full_audio).unsqueeze(0)

        # Stream response
        buffer = io.BytesIO()
        ta.save(buffer, final_waveform, sample_rate, format="wav")
        buffer.seek(0)

        headers = {"Content-Disposition": 'attachment; filename="speech.wav"'}

        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Speech generation failed. Please try again later.")