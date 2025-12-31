"""
FastAPI Whisper Transcriber Server (GPU-friendly)
- Only Whisper transcription (no LLM, no TTS, no memory)
- Accepts WAV audio (any SR), converts to float32 mono, resamples to 16k
- Returns JSON: { session_id, text, lang, duration_ms, audio_sec }
"""

import os
import io
import time
import uuid
import wave
from typing import Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

load_dotenv()

# =========================
# Config
# =========================

TARGET_SR = 16000

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-base-1m.hr-ctranslate2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")          # "cuda" on RunPod GPU
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")     # float16 for GPU speed, int8 for CPU

# If you want auth like before:
SERVER_AUTH_TOKEN = os.getenv("REMOTE_SERVER_AUTH_TOKEN", "").strip() or None

app = FastAPI(title="Whisper Transcriber Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Auth"],
)

# =========================
# Auth (optional)
# =========================

def require_auth(x_auth: Optional[str] = Header(None)):
    if SERVER_AUTH_TOKEN and x_auth != SERVER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid auth token")

# =========================
# Audio utils
# =========================

def _read_wav_bytes_to_float32_mono(wav_bytes: bytes) -> Tuple[np.ndarray, int, float]:
    """
    Read WAV bytes via stdlib wave.
    Supports PCM 16-bit / 24-bit / 32-bit (basic int32 handling).
    Returns (mono_float32, sr, audio_seconds).
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (b[:, 0].astype(np.int32) |
                  (b[:, 1].astype(np.int32) << 8) |
                  (b[:, 2].astype(np.int32) << 16))
        signed = (signed << 8) >> 8  # sign-extend 24->32
        x = signed.astype(np.float32) / 8388608.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported WAV sampwidth={sw}")

    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)

    audio_sec = float(len(x)) / float(sr) if sr else 0.0
    return x, sr, audio_sec


def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio.astype(np.float32, copy=False)
    target_len = int(len(audio) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    from scipy.signal import resample
    return resample(audio, target_len).astype(np.float32)

# =========================
# Whisper
# =========================

def load_whisper() -> WhisperModel:
    kwargs = dict(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        cpu_threads=os.cpu_count() or 4,
        num_workers=4,
    )
    model = WhisperModel(WHISPER_MODEL, **kwargs)

    # Optional warmup for CUDA
    if WHISPER_DEVICE == "cuda":
        print("Warming up Whisper on GPU...")
        dummy = np.zeros(TARGET_SR, dtype=np.float32)
        try:
            segments, _info = model.transcribe(
                audio=dummy,
                beam_size=1,
                vad_filter=False,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            # Force iteration
            _ = "".join(seg.text for seg in segments)
            print("✓ Warmup done")
        except Exception as e:
            print(f"Warmup warning: {e}")

    return model

whisper_model = load_whisper()

def whisper_transcribe(audio_16k: np.ndarray):
    segments, info = whisper_model.transcribe(
        audio=audio_16k,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        language=None,  # auto detect
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", None)
    return text, lang

# =========================
# API
# =========================

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE,
    }

@app.post("/api/session")
def create_session(_: None = Depends(require_auth)):
    return {"session_id": uuid.uuid4().hex}

@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    # Keep interface compatible with your existing client
    session_id = session_id or uuid.uuid4().hex

    # Decode WAV -> float32 mono -> 16k
    try:
        x, sr, audio_sec = _read_wav_bytes_to_float32_mono(audio_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV: {e}")

    if len(x) < int(0.2 * sr):  # too short
        return {
            "session_id": session_id,
            "text": "",
            "lang": None,
            "assistant_text": "",      # keep for client compatibility
            "tts_audio_b64": None,     # keep for client compatibility
            "tts_sample_rate": TARGET_SR,
            "duration_ms": 0.0,
            "audio_sec": audio_sec,
            "skipped": True,
            "reason": "too_short",
        }

    x16 = _resample_to_16k(x, sr)

    t0 = time.perf_counter()
    text, lang = whisper_transcribe(x16)
    t1 = time.perf_counter()

    return {
        "session_id": session_id,
        "text": text,
        "lang": lang,
        "assistant_text": "",      # client expects it (leave empty)
        "tts_audio_b64": None,     # no TTS
        "tts_sample_rate": TARGET_SR,
        "duration_ms": (t1 - t0) * 1000.0,
        "audio_sec": audio_sec,
        "skipped": False if text else True,
        "reason": None if text else "empty_transcript",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Whisper-only server on :{port}")
    print(f"Model: {WHISPER_MODEL} | device={WHISPER_DEVICE} | compute={WHISPER_COMPUTE}")
    uvicorn.run(app, host="0.0.0.0", port=port)
