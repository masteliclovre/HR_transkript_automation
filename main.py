"""
# Remote-only voice agent - WAV direct insert
# - Loads a local WAV file
# - Sends it to remote agent /api/process
# - Plays returned TTS audio (PCM16)
"""

import os
import io
import sys
import time
import wave
import base64
import contextlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    raise SystemExit(1) from exc

from dotenv import load_dotenv
from scipy.signal import resample
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# =========================
# Configuration
# =========================

REMOTE_AGENT_URL = os.getenv("REMOTE_AGENT_URL", "").strip() or None
REMOTE_AGENT_TOKEN = os.getenv("REMOTE_AGENT_TOKEN", "").strip() or None
REMOTE_AGENT_OPENAI_KEY = os.getenv("REMOTE_AGENT_OPENAI_KEY", "").strip() or None

TARGET_SR = 16000

HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "4.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "60.0"))
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)


# =========================
# HTTP Session
# =========================

def _configure_http_session(session: requests.Session) -> requests.Session:
    adapter = HTTPAdapter(
        pool_connections=8,
        pool_maxsize=16,
        max_retries=Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 502, 503, 504],
        ),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =========================
# Audio Output
# =========================

class OutputAudio:
    def __init__(self, samplerate=TARGET_SR, channels=1):
        stream_kwargs = dict(samplerate=samplerate, channels=channels, dtype="float32")
        try:
            self.stream = sd.OutputStream(latency="low", **stream_kwargs)
        except Exception:
            self.stream = sd.OutputStream(**stream_kwargs)
        self.stream.start()

    def write_int16_bytes(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.write(pcm.reshape(-1, 1))

    def write_float_np(self, pcm: np.ndarray):
        if pcm.size == 0:
            return
        self.stream.write(pcm.reshape(-1, 1))

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# =========================
# Audio Utilities
# =========================

def _read_wav_to_float32_mono(path: str) -> tuple[np.ndarray, int]:
    """
    Read WAV using stdlib wave.
    Supports PCM 16-bit / 24-bit / 32-bit and float32 (basic handling).
    Returns mono float32 in [-1,1] and sample rate.
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # Could be int32 PCM; treat as int32
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 3:
        # 24-bit PCM -> unpack manually
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (b[:, 0].astype(np.int32) |
                  (b[:, 1].astype(np.int32) << 8) |
                  (b[:, 2].astype(np.int32) << 16))
        signed = (signed << 8) >> 8  # sign-extend 24->32
        x = signed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported WAV sampwidth={sampwidth} bytes")

    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)

    return x, sr


def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np.astype(np.float32)
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)


def float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    audio_16k = resample_to_16k(audio_np, sr)
    int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    return buf


# =========================
# Remote Agent Client
# =========================

@dataclass
class RemoteAgentResult:
    user_text: str
    assistant_text: str
    lang: Optional[str]
    audio_pcm16: bytes
    sample_rate: int
    session_id: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None


class RemoteAgentClient:
    def __init__(self, base_url: str, token: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api"):
            self.base_url = f"{self.base_url}/api"
        self.token = token.strip() if token else None
        self.openai_api_key = openai_api_key.strip() if openai_api_key else None
        self.session_id: Optional[str] = None
        self.session = _configure_http_session(requests.Session())

    def _headers(self):
        headers = {}
        if self.token:
            headers["X-Auth"] = self.token
        if self.openai_api_key:
            headers["X-OpenAI-Key"] = self.openai_api_key
        return headers

    def ensure_session(self):
        if self.session_id:
            return
        resp = self.session.post(f"{self.base_url}/session", headers=self._headers(), timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        sid = data.get("session_id")
        if not sid:
            raise RuntimeError("Remote agent did not return a session_id")
        self.session_id = sid

    def process(self, wav_buf: io.BytesIO) -> Optional[RemoteAgentResult]:
        payload = wav_buf.getvalue()
        if not payload:
            return None

        attempts = 0
        while attempts < 2:
            attempts += 1
            self.ensure_session()

            files = {"audio": ("audio.wav", payload, "audio/wav")}
            data = {"session_id": self.session_id}
            if self.openai_api_key:
                data["openai_api_key"] = self.openai_api_key

            resp = self.session.post(
                f"{self.base_url}/process",
                headers=self._headers(),
                files=files,
                data=data,
                timeout=HTTP_TIMEOUT,
            )

            if resp.status_code in (401, 403):
                raise RuntimeError("Remote agent authentication failed")
            if resp.status_code in (404, 410):
                self.session_id = None
                continue

            resp.raise_for_status()
            body = resp.json()

            if body.get("error"):
                raise RuntimeError(body["error"])

            sid = body.get("session_id")
            if sid:
                self.session_id = sid

            user_text = body.get("text", "") or ""
            assistant_text = body.get("assistant_text", "") or ""
            lang = body.get("lang") or None
            audio_b64 = body.get("tts_audio_b64")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            sr = int(body.get("tts_sample_rate", TARGET_SR) or TARGET_SR)
            skipped = bool(body.get("skipped"))

            return RemoteAgentResult(
                user_text=user_text,
                assistant_text=assistant_text,
                lang=lang,
                audio_pcm16=audio_bytes,
                sample_rate=sr,
                session_id=self.session_id,
                skipped=skipped,
                reason=body.get("reason"),
            )

        raise RuntimeError("Remote agent unavailable after retries")


# =========================
# Main
# =========================

def main():
    if not REMOTE_AGENT_URL:
        print("❌ ERROR: REMOTE_AGENT_URL is not set (set it in .env)")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python wav_remote_agent.py /path/to/input.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.isfile(wav_path):
        print(f"❌ File not found: {wav_path}")
        sys.exit(1)

    print(f"✅ Remote backend: {REMOTE_AGENT_URL}")
    print(f"📄 Input WAV: {wav_path}")

    audio, sr = _read_wav_to_float32_mono(wav_path)
    wav_buf = float32_to_wav_bytes(audio, sr)

    client = RemoteAgentClient(
        REMOTE_AGENT_URL,
        REMOTE_AGENT_TOKEN,
        openai_api_key=REMOTE_AGENT_OPENAI_KEY,
    )

    out = OutputAudio(samplerate=TARGET_SR, channels=1)
    try:
        t0 = time.perf_counter()
        result = client.process(wav_buf)
        t1 = time.perf_counter()

        if not result:
            print("⚠️ No result returned.")
            return

        flag = "🇭🇷" if (result.lang or "").startswith("hr") else "🇬🇧"
        print(f"{flag} You: {result.user_text}")
        print(f"🤖 Assistant: {result.assistant_text}")
        print(f"⏱️ Remote total: {(t1 - t0)*1000:.1f}ms")

        if result.audio_pcm16:
            if result.sample_rate != TARGET_SR:
                pcm = np.frombuffer(result.audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
                pcm_16k = resample_to_16k(pcm, result.sample_rate)
                out.write_float_np(pcm_16k)
            else:
                out.write_int16_bytes(result.audio_pcm16)
        else:
            print("⚠️ Remote agent returned no audio.")
    finally:
        out.close()


if __name__ == "__main__":
    main()
