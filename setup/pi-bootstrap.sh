#!/usr/bin/env bash
#
# Rebuild the Whisplay AI chatbot on a fresh Raspberry Pi OS Lite (64-bit,
# Debian 13 "trixie") install on a Pi Zero 2 W with the PiSugar Whisplay HAT
# (WM8960 audio). See setup/README.md for flashing the card first.
#
# Run as the normal user (not root); it uses sudo where needed:
#   curl -fsSL https://raw.githubusercontent.com/ML-flash/whisplay-ai-chatbot/master/setup/pi-bootstrap.sh -o pi-bootstrap.sh
#   LLM_HOST=172.16.101.249 bash pi-bootstrap.sh
#
# Safe to re-run: every step checks what is already done. Takes roughly
# 30-60 minutes on a Pi Zero 2 W (yarn install and emoji pre-rendering are
# the slow parts). Reboot when it finishes.
set -Eeo pipefail
trap 'echo "[X] failed at line $LINENO" >&2' ERR

LLM_HOST="${LLM_HOST:-172.16.101.249}"      # PC running LM Studio
LLM_MODEL="${LLM_MODEL:-qwen/qwen3-30b-a3b-2507}"  # fallback when none is loaded
REPO_URL="${REPO_URL:-https://github.com/ML-flash/whisplay-ai-chatbot.git}"
APP="$HOME/whisplay-ai-chatbot"
VOSK_DIR="$HOME/vosk"
VOSK_MODEL="vosk-model-small-en-us-0.15"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium"
NODE_MAJOR=20

log() { echo; echo "==== $*"; }
[[ $EUID -ne 0 ]] || { echo "Run as your normal user, not root." >&2; exit 1; }
grep -q "Raspberry Pi" /proc/device-tree/model || { echo "Not a Raspberry Pi." >&2; exit 1; }

log "1/11 system packages"
sudo apt-get update
sudo apt-get install -y git curl unzip ffmpeg sox libsox-fmt-mp3 mpg123 alsa-utils \
    python3-pip python3-dev python3-pil python3-spidev python3-rpi-lgpio python3-libgpiod \
    python3-cairosvg libcairo2 logrotate

log "2/11 Whisplay HAT driver (legacy WM8960 branch, the one this chatbot expects)"
if [[ ! -d "$HOME/Whisplay" ]]; then
    git clone --depth 1 -b support/wm8960 https://github.com/PiSugar/Whisplay.git "$HOME/Whisplay"
fi
if ! systemctl is-enabled --quiet wm8960-soundcard.service 2>/dev/null; then
    # the installer asks "Proceed? [y/N]" once (printf, not `yes`: with
    # pipefail, `yes` dying of SIGPIPE would fail this step)
    (cd "$HOME/Whisplay" && printf 'y\n' | sudo bash script/install_raspberry_pi.sh)
fi

log "3/11 Node.js $NODE_MAJOR (nvm) and yarn"
export NVM_DIR="$HOME/.nvm"
if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm install "$NODE_MAJOR"
nvm alias default "$NODE_MAJOR"
command -v yarn >/dev/null || npm install -g yarn

log "4/11 chatbot code"
if [[ ! -d "$APP/.git" ]]; then
    git clone "$REPO_URL" "$APP"
else
    git -C "$APP" pull --ff-only
fi
cd "$APP"

log "5/11 display assets (font, emoji)"
cd "$APP/python"
[[ -f NotoSansSC-Bold.ttf ]] || curl -fsSL -o NotoSansSC-Bold.ttf https://cdn.pisugar.com/EchoView/NotoSansSC-Bold.ttf
if [[ ! -d emoji_svg ]]; then
    curl -fsSL -o emoji_svg.zip https://cdn.pisugar.com/EchoView/emoji_svg.zip
    unzip -oq emoji_svg.zip && rm -f emoji_svg.zip
fi
cd "$APP"

log "6/11 Node dependencies and build"
yarn install --frozen-lockfile
rm -rf dist
node node_modules/typescript/bin/tsc

log "7/11 speech-to-text: Vosk + small English model"
python3 -c "import vosk" 2>/dev/null || pip3 install --user --break-system-packages vosk
mkdir -p "$VOSK_DIR"
if [[ ! -d "$VOSK_DIR/$VOSK_MODEL" ]]; then
    curl -fsSL -o "$VOSK_DIR/model.zip" "https://alphacephei.com/vosk/models/$VOSK_MODEL.zip"
    (cd "$VOSK_DIR" && unzip -q model.zip && rm -f model.zip)
fi

log "8/11 text-to-speech: Piper + Amy voice"
mkdir -p "$APP/piper_tts"
cd "$APP/piper_tts"
if [[ ! -x piper/piper ]]; then
    curl -fsSL -o piper.tar.gz "$PIPER_URL"
    tar xzf piper.tar.gz && rm -f piper.tar.gz
fi
for f in en_US-amy-medium.onnx en_US-amy-medium.onnx.json; do
    [[ -s "$f" ]] || curl -fsSL -o "$f" "$VOICE_URL/$f"
done
cd "$APP"

log "9/11 pre-render emoji (one-off, ~6 minutes)"
if [[ ! -d python/emoji_png/40 ]]; then
    (cd python && nice -n 19 python3 -c "from utils import EmojiUtils; EmojiUtils.prerender_all((20, 24, 40))")
fi

log "10/11 .env"
if [[ ! -f .env ]]; then
    cat > .env <<EOF
ASR_SERVER=vosk
LLM_SERVER=lmstudio
TTS_SERVER=piper
ENABLE_THINKING=false

LMSTUDIO_BASE_URL=http://$LLM_HOST:1234/v1
LMSTUDIO_MODEL=$LLM_MODEL
LMSTUDIO_THINKING=auto

VOSK_MODEL_PATH=$VOSK_DIR/$VOSK_MODEL
PIPER_BINARY_PATH=$APP/piper_tts/piper/piper
PIPER_MODEL_PATH=$APP/piper_tts/en_US-amy-medium.onnx

# seconds of idle before the screen turns off (0 = immediately, -1 = never)
# SCREEN_IDLE_TIMEOUT=10
EOF
else
    echo ".env exists, leaving it unchanged"
fi

log "11/11 system tuning, SD-card protection and the chatbot service"
BOOT=/boot/firmware/config.txt
# headless: no KMS graphics driver (frees ~190MB of CMA), minimum GPU memory,
# no onboard HDMI audio (the WM8960 becomes card 0), no camera/display probing
sudo sed -i -e 's/^dtoverlay=vc4-kms-v3d/#dtoverlay=vc4-kms-v3d/' \
            -e 's/^max_framebuffers=2/#max_framebuffers=2/' \
            -e 's/^camera_auto_detect=1/camera_auto_detect=0/' \
            -e 's/^display_auto_detect=1/display_auto_detect=0/' \
            -e 's/^dtparam=audio=on/dtparam=audio=off/' "$BOOT"
grep -q '^gpu_mem=' "$BOOT" || printf '\n[all]\n# headless chatbot: minimum GPU memory\ngpu_mem=16\n' | sudo tee -a "$BOOT" >/dev/null

# services a headless chatbot does not need (absent ones are ignored)
for unit in bluetooth.service hciuart.service ModemManager.service cups.service cups.socket \
            cups.path cups-browsed.service rpcbind.service rpcbind.socket nfs-blkmap.service; do
    sudo systemctl disable --now "$unit" >/dev/null 2>&1 || true
done

# SD-card wear: keep the system journal in RAM and rotate the chatbot log
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=16M\n' | sudo tee /etc/systemd/journald.conf.d/volatile.conf >/dev/null
sudo tee /etc/logrotate.d/whisplay-chatbot >/dev/null <<EOF
$APP/chatbot.log {
    size 5M
    rotate 2
    copytruncate
    missingok
    notifempty
    compress
}
EOF

sudo tee /etc/systemd/system/chatbot.service >/dev/null <<EOF
[Unit]
Description=Whisplay AI Chatbot
After=network-online.target sound.target wm8960-soundcard.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=audio
SupplementaryGroups=audio spi gpio i2c video
WorkingDirectory=$APP
ExecStart=/bin/bash $APP/run_chatbot.sh
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=HOME=$HOME
Environment=XDG_RUNTIME_DIR=/run/user/$(id -u)
PrivateDevices=no
StandardOutput=append:$APP/chatbot.log
StandardError=append:$APP/chatbot.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable chatbot.service
sudo systemctl set-default multi-user.target

log "done - reboot to apply the boot and driver changes:  sudo reboot"
