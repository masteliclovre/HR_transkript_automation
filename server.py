# server.py
import os
import time
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-base-1m.hr-ctranslate2")

# device = "cuda" za GPU; fallback na CPU ako nema GPU-a (ali ti želiš GPU)
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")          # "cuda" | "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # "float16" je tipično za GPU
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))

app = FastAPI(title="HR Whisper Transcription API")

model: Optional[WhisperModel] = None


@app.on_event("startup")
def load_model():
    global model
    # Napomena: ako CUDA nije dostupna, faster-whisper će baciti error.
    # Po želji možeš wrapati try/except i fallback na CPU.
    model = WhisperModel(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )


@app.get("/health")
def health():
    return {"ok": True, "model": WHISPER_MODEL, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "hr",           # forsiraj hrvatski
    vad_filter: bool = True,        # VAD pomaže kod tišine
    beam_size: int = 5,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.lower().endswith(".wav"):
        # možeš ovo maknut ako želiš podržat i mp3/m4a; faster-whisper može čitati puno formata
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    start = time.time()

    # Spremi upload u privremenu datoteku (faster-whisper čita s patha najstabilnije)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {e}")

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            word_timestamps=False,  # stavi True ako želiš riječi + vrijeme
        )

        seg_list = []
        full_text_parts = []
        for seg in segments:
            seg_list.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
            )
            full_text_parts.append(seg.text.strip())

        elapsed = time.time() - start

        return JSONResponse(
            {
                "text": " ".join([t for t in full_text_parts if t]),
                "segments": seg_list,
                "language": info.language,
                "language_probability": float(info.language_probability),
                "duration": float(getattr(info, "duration", 0.0)),
                "time_sec": round(elapsed, 3),
                "model": WHISPER_MODEL,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
