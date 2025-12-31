# main.py
import argparse
import requests
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/transcribe", help="API endpoint")
    parser.add_argument("--wav", required=True, help="Path to .wav file")
    parser.add_argument("--language", default="hr", help="Language code (default: hr)")
    parser.add_argument("--vad", action="store_true", help="Enable VAD filter")
    parser.add_argument("--beam", type=int, default=5, help="Beam size")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise SystemExit(f"File not found: {wav_path}")

    with wav_path.open("rb") as f:
        files = {"file": (wav_path.name, f, "audio/wav")}
        data = {
            "language": args.language,
            "vad_filter": str(args.vad).lower(),
            "beam_size": str(args.beam),
        }
        r = requests.post(args.url, files=files, data=data, timeout=600)

    if r.status_code != 200:
        raise SystemExit(f"Error {r.status_code}: {r.text}")

    payload = r.json()

    print("\n=== TEXT ===\n")
    print(payload.get("text", ""))

    print("\n=== SEGMENTS ===\n")
    for s in payload.get("segments", []):
        print(f"[{s['start']:.2f} -> {s['end']:.2f}] {s['text']}")

if __name__ == "__main__":
    main()
