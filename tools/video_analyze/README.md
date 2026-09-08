# video_analyze

ショート動画を「読める素材」に分解するツール。カット検出・コンタクトシート・音声区間・日本語文字起こしを一括で出す。

```bash
pip install -r requirements.txt
bash setup_models.sh                     # 初回のみ。VAD + SenseVoice ASR（約230MB）
python3 analyze.py video.mp4 --lang ja   # -> video_analysis/report.md ほか
```

外部 API を使わない（すべてローカル）。ffmpeg は imageio-ffmpeg 同梱のバイナリを使うのでシステムに ffmpeg が無くても動く。

分析の観点・手順は `.claude/skills/video-analyze/SKILL.md` を参照。

## YouTube URL を直接見る（Gemini 経由）

ローカルにファイルが無いとき、Gemini API に URL を渡して観察記録（逐語文字起こし・テロップ・ショット一覧、全部タイムコード付き）を取る。

```bash
export GEMINI_API_KEY=...                       # https://aistudio.google.com/apikey
python3 gemini_video.py "https://youtube.com/shorts/XXXX" --out out_dir
python3 gemini_video.py --list-models           # モデル名の確認（既定: gemini-2.5-pro、GEMINI_MODEL で変更）
```

ローカルファイルも渡せる（20MB 以下はインライン、それ以上は Files API 経由）。依存ライブラリなし。
