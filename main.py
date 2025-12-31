# main_split.py
import argparse
import glob
import subprocess
import time
from pathlib import Path

import requests


# =========================
# Audio split (ffmpeg)
# =========================
def ffmpeg_split(input_wav: Path, segment_sec: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "chunk_%03d.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(segment_sec),
        str(out_pattern),
    ]
    subprocess.check_call(cmd)

    return sorted(out_dir.glob("chunk_*.wav"))


# =========================
# POST single chunk (with retry)
# =========================
def post_chunk(
    url: str,
    wav_path: Path,
    language: str,
    vad: bool,
    beam: int,
    timeout_sec: int,
    retries: int = 5,
):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with wav_path.open("rb") as f:
                files = {"file": (wav_path.name, f, "audio/wav")}
                data = {
                    "language": language,
                    "vad_filter": str(vad).lower(),
                    "beam_size": str(beam),
                }

                r = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=(30, timeout_sec),
                )

            if r.status_code in (502, 503, 504):
                time.sleep(2 * attempt)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_error = e
            time.sleep(2 * attempt)

    raise last_error


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser(description="Large WAV → chunked Whisper transcription")
    p.add_argument("--url", required=True)
    p.add_argument("--wav", required=True, help="Input WAV or glob (chunks/*.wav)")
    p.add_argument("--segment-sec", type=int, default=180)
    p.add_argument("--out-dir", default="chunks")
    p.add_argument("--language", default="hr")
    p.add_argument("--vad", action="store_true")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--out", default="transcript.txt")
    p.add_argument("--out-segments", default="transcript_with_timestamps.txt")
    args = p.parse_args()

    # -------------------------
    # Collect chunks
    # -------------------------
    if any(ch in args.wav for ch in ["*", "?", "[", "]"]):
        chunks = sorted(Path(p).resolve() for p in glob.glob(args.wav))
        if not chunks:
            raise SystemExit(f"No files matched: {args.wav}")
    else:
        input_wav = Path(args.wav).resolve()
        if not input_wav.exists():
            raise SystemExit(f"File not found: {input_wav}")
        chunks = ffmpeg_split(input_wav, args.segment_sec, Path(args.out_dir))

    # -------------------------
    # Transcribe
    # -------------------------
    full_text: list[str] = []
    all_segments: list[dict] = []
    offset = 0.0

    for i, wav_path in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] {wav_path.name}")

        payload = post_chunk(
            args.url,
            wav_path,
            args.language,
            args.vad,
            args.beam,
            args.timeout,
        )

        text = payload.get("text", "").strip()
        if text:
            full_text.append(text)

        segs = payload.get("segments", [])
        for s in segs:
            all_segments.append({
                "start": float(s["start"]) + offset,
                "end": float(s["end"]) + offset,
                "text": s["text"],
            })

        dur = float(payload.get("duration") or 0.0)
        if dur <= 0.0 and segs:
            dur = float(segs[-1]["end"])
        offset += dur

    # -------------------------
    # Save outputs
    # -------------------------
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))

    with open(args.out_segments, "w", encoding="utf-8") as f:
        for s in all_segments:
            f.write(f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}\n")

    print(f"\n✅ TXT saved: {args.out}")
    print(f"🕒 Segments saved: {args.out_segments}")


if __name__ == "__main__":
    main()
