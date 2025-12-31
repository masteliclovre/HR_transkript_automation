# main.py
import argparse
import subprocess
import time
from pathlib import Path
from textwrap import wrap

import requests


# =========================
# Time formatting
# =========================
def format_srt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# =========================
# Text helpers
# =========================
def clean_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def vtt_wrap_lines(text: str, width: int = 42, max_lines: int = 2) -> str:
    text = clean_text(text)
    if not text:
        return ""

    lines = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= max_lines:
        return "\n".join(lines)

    kept = lines[: max_lines - 1]
    last = " ".join(lines[max_lines - 1 :])
    last_wrapped = wrap(
        last,
        width=int(width * 1.2),
        break_long_words=False,
        break_on_hyphens=False,
    )
    kept.append(last_wrapped[0] if last_wrapped else last)
    return "\n".join(kept)


# =========================
# Subtitle block builder
# =========================
def build_subtitle_blocks(
    segments: list[dict],
    max_block_sec: float,
    min_block_sec: float,
    max_chars: int,
    join_gap_sec: float,
) -> list[dict]:
    out = []
    cur = None

    def flush():
        nonlocal cur
        if cur and clean_text(cur["text"]):
            out.append(
                {
                    "start": cur["start"],
                    "end": cur["end"],
                    "text": clean_text(cur["text"]),
                }
            )
        cur = None

    for seg in segments:
        t = clean_text(seg["text"])
        if not t:
            continue

        s_start = float(seg["start"])
        s_end = float(seg["end"])

        if cur is None:
            cur = {"start": s_start, "end": s_end, "text": t}
            continue

        gap = s_start - cur["end"]
        proposed_text = f"{cur['text']} {t}".strip()
        proposed_dur = s_end - cur["start"]

        if (
            gap > join_gap_sec
            or proposed_dur > max_block_sec
            or len(proposed_text) > max_chars
        ):
            flush()
            cur = {"start": s_start, "end": s_end, "text": t}
        else:
            cur["end"] = s_end
            cur["text"] = proposed_text

    flush()

    merged = []
    i = 0
    while i < len(out):
        b = out[i]
        dur = b["end"] - b["start"]
        if dur < min_block_sec and i + 1 < len(out):
            nxt = out[i + 1]
            gap = nxt["start"] - b["end"]
            proposed_text = f"{b['text']} {nxt['text']}".strip()
            proposed_dur = nxt["end"] - b["start"]
            if (
                gap <= join_gap_sec
                and proposed_dur <= max_block_sec
                and len(proposed_text) <= max_chars
            ):
                merged.append(
                    {
                        "start": b["start"],
                        "end": nxt["end"],
                        "text": proposed_text,
                    }
                )
                i += 2
                continue
        merged.append(b)
        i += 1

    return merged


# =========================
# Audio split (ffmpeg)
# =========================
def ffmpeg_split(input_wav: Path, segment_sec: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "chunk_%03d.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(segment_sec),
        str(out_pattern),
    ]
    subprocess.check_call(cmd)
    return sorted(out_dir.glob("chunk_*.wav"))


# =========================
# POST chunk (with retry)
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
    p = argparse.ArgumentParser(description="Podcast transcription (TXT + SRT + clean VTT)")
    p.add_argument("--url", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--preset", choices=["youtube"], help="Apply preset (youtube)")
    p.add_argument("--segment-sec", type=int, default=180)
    p.add_argument("--out-dir", default="chunks")
    p.add_argument("--language", default="hr")
    p.add_argument("--vad", action="store_true")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1200)

    p.add_argument("--out-txt", default="transcript.txt")
    p.add_argument("--out-segments", default="transcript_with_timestamps.txt")
    p.add_argument("--out-srt", default="transcript.srt")
    p.add_argument("--out-vtt", default="transcript.vtt")

    p.add_argument("--blocks", action="store_true")
    p.add_argument("--max-block-sec", type=float, default=6.0)
    p.add_argument("--min-block-sec", type=float, default=1.0)
    p.add_argument("--max-chars", type=int, default=120)
    p.add_argument("--join-gap-sec", type=float, default=0.6)

    p.add_argument("--vtt-width", type=int, default=42)
    p.add_argument("--vtt-max-lines", type=int, default=2)

    args = p.parse_args()

    # -------------------------
    # Preset: YouTube podcast
    # -------------------------
    if args.preset == "youtube":
        args.blocks = True
        args.segment_sec = 180
        args.beam = 1
        args.max_block_sec = 6.5
        args.join_gap_sec = 0.7
        args.max_chars = 120
        args.vtt_width = 38
        args.vtt_max_lines = 2

    input_wav = Path(args.wav).resolve()
    if not input_wav.exists():
        raise SystemExit(f"File not found: {input_wav}")

    chunks = ffmpeg_split(input_wav, args.segment_sec, Path(args.out_dir))

    full_text = []
    raw_segments = []
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

        text = clean_text(payload.get("text", ""))
        if text:
            full_text.append(text)

        segs = payload.get("segments", [])
        for s in segs:
            seg_text = clean_text(s.get("text", ""))
            if not seg_text:
                continue
            raw_segments.append(
                {
                    "start": float(s["start"]) + offset,
                    "end": float(s["end"]) + offset,
                    "text": seg_text,
                }
            )

        dur = float(payload.get("duration") or 0.0)
        if dur <= 0.0 and segs:
            dur = float(segs[-1]["end"])
        offset += dur

    subtitle_units = (
        build_subtitle_blocks(
            raw_segments,
            args.max_block_sec,
            args.min_block_sec,
            args.max_chars,
            args.join_gap_sec,
        )
        if args.blocks
        else raw_segments
    )

    # TXT
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))

    # Timestamp TXT
    with open(args.out_segments, "w", encoding="utf-8") as f:
        for s in subtitle_units:
            f.write(f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}\n")

    # SRT
    with open(args.out_srt, "w", encoding="utf-8") as f:
        for i, s in enumerate(subtitle_units, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n"
            )
            f.write(f"{s['text']}\n\n")

    # Clean VTT
    with open(args.out_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in subtitle_units:
            cue_text = vtt_wrap_lines(
                s["text"],
                width=args.vtt_width,
                max_lines=args.vtt_max_lines,
            )
            if not cue_text:
                continue
            f.write(
                f"{format_vtt_time(s['start'])} --> {format_vtt_time(s['end'])}\n"
            )
            f.write(f"{cue_text}\n\n")

    print("\n✅ DONE")
    print(f"🎬 SRT: {args.out_srt}")
    print(f"🌐 VTT: {args.out_vtt}")
    print(f"📄 TXT: {args.out_txt}")


if __name__ == "__main__":
    main()
