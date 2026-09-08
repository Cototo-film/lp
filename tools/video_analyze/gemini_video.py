#!/usr/bin/env python3
"""
Gemini as the "eyes": send a YouTube URL (or a local video file) to the Gemini API
and get back a timecoded OBSERVATION of the video (transcript, on-screen text,
shot list, visual events). Analysis is done afterwards by Claude with the
video-analyze skill, on top of these observations.

Why two passes: Gemini watches the video, which this environment cannot fetch.
Keeping observation (facts) separate from analysis (judgement) lets the
analysis be checked against the facts instead of being taken on trust.

Usage:
  export GEMINI_API_KEY=...            # https://aistudio.google.com/apikey
  python3 gemini_video.py URL_OR_FILE [--out DIR] [--model MODEL] [--mode observe|analyze|both]
  python3 gemini_video.py --list-models

No third-party dependencies (urllib only). Honors HTTPS_PROXY / SSL_CERT_FILE.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # Flash works on the free tier; Pro models need Cloud billing
INLINE_LIMIT = 19 * 1024 * 1024   # inline base64 limit is 20MB total request

OBSERVE_PROMPT = """あなたは映像の観察係です。判断や評価はせず、動画に「実際に含まれているもの」だけを、タイムコード付きで書き出してください。推測は書かない。聞き取れない・読めない箇所は [不明] と書く。

出力は次の Markdown 構成で、見出し名を変えないこと。

## 基本情報
- 尺（秒）、縦横、言語、ナレーションの有無（人の声か合成音声かは「判断できる根拠」と一緒に）

## 文字起こし（逐語）
- `[mm:ss] 発話` の形式で、言い直しや間投詞も含めて逐語で。

## 画面上のテキスト（テロップ・図中の文字）
- `[mm:ss] 位置（上/中/下） 「テキスト」` の形式で、出た順に。強調（色・サイズ）があれば併記。

## ショット一覧
- `[mm:ss–mm:ss] 何が映っているか（被写体・構図・動き・カメラワーク）` を、カットごとに。ディゾルブや同一ショット内のズームは「同一ショット」と明記。

## 視覚イベント
- 矢印、数字、ハイライト、図解、ビフォーアフター、指差しなど、目を止める要素をタイムコード付きで。

## 音
- BGM の有無と印象ではなく種類（テンポ・音量の変化点）、効果音、無音区間。

## 終わり方
- 最後の3秒に何があるか（CTA の文言、ロゴ、次の動画への誘導）。
"""

ANALYZE_PROMPT = """以下の観察記録（事実）だけを根拠に、この動画が「分かりやすい」または「伸びる」理由を分析してください。観察記録にない事実（チャンネルの出自、再生数、制作ツール）は書かない。印象語（テンポがいい、引き込まれる）だけの記述は禁止。各主張にタイムコードを添える。

観点:
1. 冒頭2秒の設計（フック）
2. 主張の構造（結論→理由→例 / 問題→解決 / 比較 / 手順）と転換点
3. 音声・テロップ・映像の一致（三重符号化）と、その不一致
4. カットとリズム（ショット長の分布、情報の切り替わりとカットの一致）
5. 視覚の錨（数字・矢印・図解など）
6. 認知負荷（1文の長さ、専門用語、言い換え）
7. 離脱リスクと CTA

出力: 結論1段落 → 観点ごとの根拠表（観点 | 根拠(タイムコード) | 評価） → 転用できる技術3つ / 文脈依存で真似すべきでない点。
"""


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        sys.exit("GEMINI_API_KEY is not set. Get one at https://aistudio.google.com/apikey and export it.")
    return k


def _request(method: str, url: str, body: dict | bytes | None = None, headers: dict | None = None,
             timeout: int = 600) -> dict:
    data = None
    h = {"x-goog-api-key": _key()}
    if headers:
        h.update(headers)
    if isinstance(body, dict):
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    elif isinstance(body, bytes):
        data = body
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        sys.exit(f"Gemini API {e.code} on {url.split('?')[0]}:\n{msg[:2000]}")


def list_models() -> None:
    d = _request("GET", f"{API}/v1beta/models?pageSize=100")
    for m in d.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(f"{m['name'].removeprefix('models/'):40s} {m.get('displayName','')}")


def upload_file(path: Path) -> str:
    """Resumable upload via the Files API; returns the file URI once ACTIVE."""
    mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
    size = path.stat().st_size
    # start call is done by hand because the upload URL comes back in a response header
    req = urllib.request.Request(f"{API}/upload/v1beta/files",
                                 data=json.dumps({"file": {"display_name": path.name}}).encode(),
                                 method="POST",
                                 headers={"x-goog-api-key": _key(), "Content-Type": "application/json",
                                          "X-Goog-Upload-Protocol": "resumable",
                                          "X-Goog-Upload-Command": "start",
                                          "X-Goog-Upload-Header-Content-Length": str(size),
                                          "X-Goog-Upload-Header-Content-Type": mime})
    with urllib.request.urlopen(req, timeout=120) as r:
        upload_url = r.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        sys.exit("Files API did not return an upload URL")
    info = _request("POST", upload_url, body=path.read_bytes(),
                    headers={"Content-Length": str(size),
                             "X-Goog-Upload-Offset": "0",
                             "X-Goog-Upload-Command": "upload, finalize"})
    f = info["file"]
    name, uri = f["name"], f["uri"]
    while f.get("state") == "PROCESSING":
        time.sleep(3)
        f = _request("GET", f"{API}/v1beta/{name}")
    if f.get("state") != "ACTIVE":
        sys.exit(f"file processing failed: {f}")
    return uri


def video_part(src: str) -> dict:
    if src.startswith("http://") or src.startswith("https://"):
        return {"file_data": {"file_uri": src}}          # YouTube URLs are accepted here
    p = Path(src)
    if not p.exists():
        sys.exit(f"no such file: {src}")
    mime = mimetypes.guess_type(str(p))[0] or "video/mp4"
    if p.stat().st_size <= INLINE_LIMIT:
        return {"inline_data": {"mime_type": mime, "data": base64.b64encode(p.read_bytes()).decode()}}
    return {"file_data": {"file_uri": upload_file(p), "mime_type": mime}}


def generate(model: str, parts: list[dict], temperature: float = 0.2) -> str:
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature}}
    d = _request("POST", f"{API}/v1beta/models/{model}:generateContent", body=body)
    try:
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        sys.exit(f"unexpected response: {json.dumps(d)[:2000]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="YouTube URL or local video file")
    ap.add_argument("--out", default=None, help="output directory (default: ./gemini_<id>)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["observe", "analyze", "both"], default="observe",
                    help="observe = facts only (default; Claude does the analysis). "
                         "analyze/both = also ask Gemini for its own analysis, kept in a separate file")
    ap.add_argument("--prompt-file", default=None,
                    help="use this file as the observation prompt instead of the built-in one "
                         "(e.g. a visual-craft or typography checklist); output goes to observation_<name>.md")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.source:
        ap.error("source is required")

    ident = args.source.rstrip("/").split("/")[-1].split("?")[0][:40] or "video"
    out = Path(args.out) if args.out else Path(f"gemini_{ident}")
    out.mkdir(parents=True, exist_ok=True)

    vp = video_part(args.source)
    prompt, obs_name = OBSERVE_PROMPT, "observation.md"
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        obs_name = f"observation_{Path(args.prompt_file).stem}.md"
    print(f"[observe] {args.model} <- {args.source}", file=sys.stderr)
    obs = generate(args.model, [vp, {"text": prompt}])
    (out / obs_name).write_text(obs, encoding="utf-8")
    print(f"wrote {out/obs_name}", file=sys.stderr)

    if args.mode in ("analyze", "both"):
        print(f"[analyze] {args.model}", file=sys.stderr)
        ana = generate(args.model, [{"text": ANALYZE_PROMPT + "\n\n---\n\n" + obs}], temperature=0.4)
        (out / "gemini_analysis.md").write_text(ana, encoding="utf-8")
        print(f"wrote {out/'gemini_analysis.md'}", file=sys.stderr)

    (out / "meta.json").write_text(json.dumps({"source": args.source, "model": args.model,
                                               "mode": args.mode, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                              ensure_ascii=False, indent=1))
    print(str(out))


if __name__ == "__main__":
    main()
