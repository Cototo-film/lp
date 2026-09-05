#!/usr/bin/env bash
#
# adget のセットアップ。依存を入れて ~/.local/bin/adget に置き、PATH を通す。
# 何度実行しても同じ結果になる(冪等)ように書いてある。

set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
ADGET_DIR="${ADGET_DIR:-$HOME/Movies/AdSwipe}"

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || warn "macOS 以外で実行しています。Homebrew 部分は読み替えてください。"

step "1/4  Homebrew を確認"
if command -v brew >/dev/null 2>&1; then
  ok "brew $(brew --version | head -n 1 | awk '{print $2}')"
else
  cat <<'MSG'

  Homebrew が入っていません。先に次を実行してください:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  完了したらこの install.sh をもう一度実行してください。
MSG
  die "Homebrew が必要です"
fi

step "2/4  yt-dlp と ffmpeg を導入 / 更新"
# X は仕様変更が多く、古い yt-dlp は動かなくなる。毎回最新に寄せる。
for pkg in yt-dlp ffmpeg; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    brew upgrade "$pkg" >/dev/null 2>&1
    ok "$pkg (最新)"
  else
    brew install "$pkg" || die "$pkg の導入に失敗しました"
    ok "$pkg (新規導入)"
  fi
done

step "3/4  adget を $BIN_DIR に配置"
mkdir -p "$BIN_DIR" || die "$BIN_DIR を作成できません"
chmod +x "$SRC_DIR/adget"
ln -sf "$SRC_DIR/adget" "$BIN_DIR/adget"
ok "$BIN_DIR/adget → $SRC_DIR/adget"

# PATH に ~/.local/bin が無ければ、目印付きで一度だけ ~/.zshrc に追記する。
MARKER="# added by adget installer"
SHELL_RC="$HOME/.zshrc"
case ":$PATH:" in
  *":$BIN_DIR:"*)
    ok "PATH は通っています"
    ;;
  *)
    if [ -f "$SHELL_RC" ] && grep -qF "$MARKER" "$SHELL_RC"; then
      warn "$SHELL_RC には追記済みです。新しいターミナルを開いてください。"
    else
      printf '\n%s\nexport PATH="$HOME/.local/bin:$PATH"\n' "$MARKER" >> "$SHELL_RC"
      ok "$SHELL_RC に PATH を追記しました（新しいターミナルから有効）"
    fi
    ;;
esac

step "4/4  保存先を用意"
mkdir -p "$ADGET_DIR" || die "$ADGET_DIR を作成できません"
ok "$ADGET_DIR"

cat <<MSG

────────────────────────────────────────────
セットアップ完了。

  adget "https://x.com/..."   URL を指定して保存
  adget                       クリップボードの URL を保存
  adget -h                    ヘルプ

保存先: $ADGET_DIR

ホットキー（コピー → キー一発で保存）の設定は
$SRC_DIR/README.md の「ホットキーにする」を参照。
────────────────────────────────────────────
MSG
