# video_analyze

ショート動画を「読める素材」に分解するツール。カット検出・コンタクトシート・音声区間・日本語文字起こしを一括で出す。

```bash
pip install -r requirements.txt
bash setup_models.sh                     # 初回のみ。VAD + SenseVoice ASR（約230MB）
python3 analyze.py video.mp4 --lang ja   # -> video_analysis/report.md ほか
```

外部 API を使わない（すべてローカル）。ffmpeg は imageio-ffmpeg 同梱のバイナリを使うのでシステムに ffmpeg が無くても動く。

分析の観点・手順は `.claude/skills/video-analyze/SKILL.md` を参照。
