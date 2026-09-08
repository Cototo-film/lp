#!/usr/bin/env python3
"""
Video analysis tool for short-form content (YouTube Shorts / Reels / TikTok).

Turns a video file into material that can be read and reasoned about:
  - shot list (cut detection) with per-shot length and motion score
  - contact sheets (keyframes with timecodes) for visual reading
  - per-shot keyframes at readable size
  - audio loudness curve, speech/silence segments (VAD)
  - transcript with timestamps (offline, sherpa-onnx SenseVoice; optional)
  - report.json + report.md

Usage:
  python3 analyze.py VIDEO [--out DIR] [--interval 1.0] [--no-asr]
                           [--model-dir DIR] [--lang auto|ja|en|zh|ko|yue]

Dependencies (all from PyPI):
  pip install imageio-ffmpeg opencv-python-headless numpy pillow sherpa-onnx
ASR model (optional): see setup_models.sh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv-python-headless is required: pip install opencv-python-headless")

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:  # pragma: no cover
    FFMPEG = "ffmpeg"

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.environ.get("VIDEO_ANALYZE_MODELS", HERE / "models"))


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def tc(seconds: float) -> str:
    """mm:ss.t timecode"""
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:04.1f}"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------------

@dataclass
class Shot:
    index: int
    start: float
    end: float
    duration: float
    motion: float            # mean inter-frame difference inside the shot (0..1)
    keyframe: str            # path of the representative frame (relative to out dir)


@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Report:
    source: str
    width: int
    height: int
    fps: float
    duration: float
    orientation: str
    n_shots: int
    avg_shot_len: float
    median_shot_len: float
    cuts_per_10s: float
    speech_ratio: float          # fraction of runtime that is speech (VAD)
    loudness_db: list[float]     # per 0.25s, dBFS
    shots: list[Shot]
    speech_segments: list[Segment]
    transcript: list[Segment]
    contact_sheets: list[str]
    notes: list[str]


# ----------------------------------------------------------------------------
# video pass: cut detection + sampling
# ----------------------------------------------------------------------------

def analyze_video(path: Path, out: Path, interval: float, cut_threshold: float,
                  min_shot: float) -> tuple[dict, list[Shot], list[tuple[float, np.ndarray]]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # analysis resolution
    aw = 160
    ah = max(1, int(height * aw / max(width, 1)))

    prev_small = None
    prev_hist = None
    scores: list[float] = []           # per frame cut score
    diffs: list[float] = []            # per frame motion (mean abs diff)
    samples: list[tuple[float, np.ndarray]] = []   # (t, frame BGR) at fixed interval
    frames_for_shots: dict[int, np.ndarray] = {}   # frame index -> full frame (lazy)

    next_sample_t = 0.0
    i = 0
    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        last_frame = frame
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).flatten()

        if prev_small is None:
            scores.append(0.0)
            diffs.append(0.0)
        else:
            d = float(np.mean(np.abs(gray - prev_small)))
            h = 1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            scores.append(0.6 * min(d * 4, 1.0) + 0.4 * min(max(h, 0.0), 1.0))
            diffs.append(d)
        prev_small, prev_hist = gray, hist

        if t + 1e-6 >= next_sample_t:
            samples.append((t, frame.copy()))
            next_sample_t += interval
        i += 1
    cap.release()

    n = len(scores)
    duration = n / fps if n else 0.0
    scores_a = np.array(scores)

    # adaptive threshold: local peak well above the neighbourhood median
    cuts: list[int] = []
    min_gap = int(min_shot * fps)
    win = int(fps)  # 1s window
    for k in range(1, n):
        if scores_a[k] < cut_threshold:
            continue
        lo, hi = max(0, k - win), min(n, k + win)
        local = np.median(scores_a[lo:hi])
        if scores_a[k] < max(cut_threshold, local * 3):
            continue
        if scores_a[k] < scores_a[max(0, k - 2):k + 3].max():
            continue
        if cuts and k - cuts[-1] < min_gap:
            continue
        cuts.append(k)

    boundaries = [0] + cuts + [n]
    shots: list[Shot] = []
    # second pass to grab representative frames (shot midpoints)
    cap = cv2.VideoCapture(str(path))
    key_dir = out / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    for s_idx in range(len(boundaries) - 1):
        a, b = boundaries[s_idx], boundaries[s_idx + 1]
        if b <= a:
            continue
        mid = a + max(1, (b - a) // 3)   # early third of the shot: usually the "settled" frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            frame = last_frame
        motion = float(np.mean(diffs[a + 1:b])) if b - a > 1 else 0.0
        kf = key_dir / f"shot_{s_idx + 1:03d}_{tc(a / fps).replace(':', '-')}.jpg"
        h = 640
        w = int(frame.shape[1] * h / frame.shape[0])
        cv2.imwrite(str(kf), cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        shots.append(Shot(index=s_idx + 1, start=a / fps, end=b / fps,
                          duration=(b - a) / fps, motion=round(motion, 4),
                          keyframe=str(kf.relative_to(out))))
    cap.release()

    meta = dict(width=width, height=height, fps=fps, duration=duration,
                n_frames=n, orientation="vertical" if height > width else "horizontal")
    return meta, shots, samples


# ----------------------------------------------------------------------------
# contact sheets
# ----------------------------------------------------------------------------

def contact_sheets(samples: list[tuple[float, np.ndarray]], shots: list[Shot], out: Path,
                   cols: int, rows: int, cell_h: int, transcript: list[Segment]) -> list[str]:
    if not samples:
        return []
    font = load_font(22)
    small_font = load_font(16)
    fh, fw = samples[0][1].shape[:2]
    cell_w = int(fw * cell_h / fh)
    label_h = 30
    per_sheet = cols * rows
    paths: list[str] = []
    cut_times = [s.start for s in shots[1:]]

    def words_at(t: float) -> str:
        for seg in transcript:
            if seg.start <= t < seg.end:
                return seg.text
        return ""

    for si in range(0, len(samples), per_sheet):
        chunk = samples[si:si + per_sheet]
        r = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * cell_w, r * (cell_h + label_h)), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for j, (t, frame) in enumerate(chunk):
            x = (j % cols) * cell_w
            y = (j // cols) * (cell_h + label_h)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((cell_w, cell_h))
            sheet.paste(img, (x, y + label_h))
            # label: timecode + cut marker
            is_new_shot = any(abs(t - c) < 1e-3 or (c > t - 1e-3 and c < t + 1e-3) for c in cut_times)
            label = tc(t)
            draw.rectangle([x, y, x + cell_w, y + label_h], fill=(18, 18, 18))
            draw.text((x + 6, y + 4), label, fill=(255, 255, 255), font=font)
            # shot number whose range contains t
            shot_no = next((s.index for s in shots if s.start - 1e-3 <= t < s.end), None)
            if shot_no is not None:
                draw.text((x + cell_w - 70, y + 6), f"S{shot_no}", fill=(255, 210, 0), font=small_font)
            w = words_at(t)
            if w:
                draw.rectangle([x, y + label_h + cell_h - 22, x + cell_w, y + label_h + cell_h],
                               fill=(0, 0, 0))
                draw.text((x + 4, y + label_h + cell_h - 20), w[:40], fill=(200, 255, 200), font=small_font)
        p = out / f"contact_sheet_{si // per_sheet + 1:02d}.jpg"
        sheet.save(p, quality=82)
        paths.append(str(p.relative_to(out)))
    return paths


# ----------------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------------

def extract_audio(path: Path, out: Path) -> Path | None:
    wav = out / "audio_16k.wav"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
        return None
    return wav


def read_wav(wav: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        data = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def loudness_curve(samples: np.ndarray, sr: int, step: float = 0.25) -> list[float]:
    hop = int(sr * step)
    out = []
    for i in range(0, len(samples), hop):
        chunk = samples[i:i + hop]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        out.append(round(20 * math.log10(max(rms, 1e-6)), 1))
    return out


def vad_and_asr(samples: np.ndarray, sr: int, model_dir: Path, lang: str,
                use_asr: bool) -> tuple[list[Segment], list[Segment], list[str]]:
    notes: list[str] = []
    try:
        import sherpa_onnx
    except ImportError:
        notes.append("sherpa-onnx not installed; no VAD/ASR (pip install sherpa-onnx)")
        return [], [], notes

    vad_model = model_dir / "silero_vad.onnx"
    if not vad_model.exists():
        notes.append(f"VAD model missing at {vad_model}; run setup_models.sh")
        return [], [], notes

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(vad_model)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 15.0
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    recognizer = None
    sv_dir = model_dir / "sense-voice"
    if use_asr:
        model_file = next((p for p in [sv_dir / "model.int8.onnx", sv_dir / "model.onnx"] if p.exists()), None)
        tokens = sv_dir / "tokens.txt"
        if model_file and tokens.exists():
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_file), tokens=str(tokens), num_threads=max(1, os.cpu_count() or 1),
                use_itn=True, language=lang, debug=False)
        else:
            notes.append(f"SenseVoice model missing under {sv_dir}; transcript skipped")

    window = cfg.silero_vad.window_size
    speech: list[Segment] = []
    transcript: list[Segment] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            start = seg.start / sr
            end = start + len(seg.samples) / sr
            speech.append(Segment(start=round(start, 2), end=round(end, 2)))
            if recognizer is not None:
                stream = recognizer.create_stream()
                stream.accept_waveform(sr, seg.samples)
                recognizer.decode_stream(stream)
                raw = stream.result.text.strip()
                # SenseVoice may emit rich tags like <|HAPPY|>, <|BGM|>, <|Laughter|>; keep them separately
                tags = re.findall(r"<\|([A-Za-z_]+)\|>", raw)
                text = re.sub(r"<\|[^|]*\|>", "", raw).strip()
                transcript.append(Segment(start=round(start, 2), end=round(end, 2), text=text, tags=tags))
            vad.pop()

    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        drain()
    vad.flush()
    drain()
    return speech, transcript, notes


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def write_markdown(rep: Report, out: Path) -> None:
    lines = []
    lines.append(f"# Video analysis: {Path(rep.source).name}\n")
    lines.append("| item | value |\n|---|---|")
    lines.append(f"| duration | {rep.duration:.1f}s |")
    lines.append(f"| size | {rep.width}x{rep.height} ({rep.orientation}), {rep.fps:.2f} fps |")
    lines.append(f"| shots | {rep.n_shots} |")
    lines.append(f"| avg / median shot | {rep.avg_shot_len:.2f}s / {rep.median_shot_len:.2f}s |")
    lines.append(f"| cuts per 10s | {rep.cuts_per_10s:.1f} |")
    lines.append(f"| speech ratio | {rep.speech_ratio * 100:.0f}% of runtime |")
    lines.append("")
    if rep.contact_sheets:
        lines.append("## Contact sheets (read these with vision)\n")
        for p in rep.contact_sheets:
            lines.append(f"- {p}")
        lines.append("")
    lines.append("## Timeline (shot × transcript)\n")
    lines.append("| # | start | len | motion | transcript in shot | keyframe |\n|---|---|---|---|---|---|")
    for s in rep.shots:
        words = " / ".join(t.text for t in rep.transcript if t.text and t.start < s.end and t.end > s.start)
        lines.append(f"| {s.index} | {tc(s.start)} | {s.duration:.2f}s | {s.motion:.3f} | {words} | {s.keyframe} |")
    lines.append("")
    if rep.transcript:
        lines.append("## Transcript\n")
        for t in rep.transcript:
            tag = f" `{' '.join(t.tags)}`" if t.tags else ""
            lines.append(f"- [{tc(t.start)}–{tc(t.end)}] {t.text}{tag}")
        lines.append("")
    if rep.speech_segments and not rep.transcript:
        lines.append("## Speech segments (VAD)\n")
        for sgm in rep.speech_segments:
            lines.append(f"- [{tc(sgm.start)}–{tc(sgm.end)}]")
        lines.append("")
    if rep.notes:
        lines.append("## Notes\n")
        lines += [f"- {n}" for n in rep.notes]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output directory (default: <video>_analysis)")
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval for contact sheets (s)")
    ap.add_argument("--cut-threshold", type=float, default=0.30, help="cut score threshold 0..1")
    ap.add_argument("--min-shot", type=float, default=0.25, help="minimum shot length (s)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cell-height", type=int, default=480)
    ap.add_argument("--no-asr", action="store_true", help="skip transcription (VAD still runs)")
    ap.add_argument("--lang", default="auto", help="auto|ja|en|zh|ko|yue")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"no such file: {video}")
    out = Path(args.out) if args.out else video.with_name(video.stem + "_analysis")
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    print(f"[1/4] video pass: {video.name}", file=sys.stderr)
    meta, shots, samples = analyze_video(video, out, args.interval, args.cut_threshold, args.min_shot)

    print("[2/4] audio pass", file=sys.stderr)
    wav = extract_audio(video, out)
    loud: list[float] = []
    speech: list[Segment] = []
    transcript: list[Segment] = []
    if wav is None:
        notes.append("no audio track (or ffmpeg failed); audio analysis skipped")
    else:
        pcm, sr = read_wav(wav)
        loud = loudness_curve(pcm, sr)
        print("[3/4] VAD / ASR", file=sys.stderr)
        speech, transcript, n2 = vad_and_asr(pcm, sr, Path(args.model_dir), args.lang, not args.no_asr)
        notes += n2

    print("[4/4] contact sheets + report", file=sys.stderr)
    sheets = contact_sheets(samples, shots, out, args.cols, args.rows, args.cell_height, transcript)

    durs = [s.duration for s in shots] or [meta["duration"]]
    speech_total = sum(s.end - s.start for s in speech)
    rep = Report(
        source=str(video), width=meta["width"], height=meta["height"], fps=round(meta["fps"], 3),
        duration=round(meta["duration"], 2), orientation=meta["orientation"],
        n_shots=len(shots), avg_shot_len=round(float(np.mean(durs)), 3),
        median_shot_len=round(float(np.median(durs)), 3),
        cuts_per_10s=round((len(shots) - 1) / max(meta["duration"], 1e-6) * 10, 2),
        speech_ratio=round(speech_total / max(meta["duration"], 1e-6), 3),
        loudness_db=loud, shots=shots, speech_segments=speech, transcript=transcript,
        contact_sheets=sheets, notes=notes,
    )
    (out / "report.json").write_text(json.dumps(asdict(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(rep, out)
    print(f"done -> {out}/report.md", file=sys.stderr)
    print(str(out))


if __name__ == "__main__":
    main()
