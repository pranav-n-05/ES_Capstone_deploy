#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 4 Model B (64-bit Raspberry Pi OS) for the INMP441 I2S microphone.
#   bash setup_pi.sh            # then: sudo reboot
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="$(id -un)"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARNING: $(uname -m) detected. Use 64-bit Raspberry Pi OS (aarch64); TFLite wheels are 64-bit only."
fi

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 alsa-utils

echo "==> Enable I2S and the INMP441 overlay"
CONFIG=/boot/firmware/config.txt
[ -f "$CONFIG" ] || CONFIG=/boot/config.txt
sudo cp "$CONFIG" "$CONFIG.bak.kws"
grep -q '^dtparam=i2s=on' "$CONFIG" || echo 'dtparam=i2s=on' | sudo tee -a "$CONFIG" >/dev/null
grep -q '^dtoverlay=googlevoicehat-soundcard' "$CONFIG" || echo 'dtoverlay=googlevoicehat-soundcard' | sudo tee -a "$CONFIG" >/dev/null
echo "    $CONFIG updated (backup: $CONFIG.bak.kws)"

echo "==> Python environment"
python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install --upgrade pip
"$DIR/.venv/bin/pip" install -r "$DIR/requirements.txt"
"$DIR/.venv/bin/pip" install ai-edge-litert || "$DIR/.venv/bin/pip" install tflite-runtime

echo "==> Offline check with a bundled recording"
(cd "$DIR" && .venv/bin/python kws_cli.py --wav samples/long_demo_3x_marvin.wav)

echo "==> Start web UI on boot (port 8000)"
sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$DIR|g" "$DIR/kws-web.service" | sudo tee /etc/systemd/system/kws-web.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable kws-web.service

echo
echo "Done. Reboot to activate the microphone overlay:  sudo reboot"
echo "After reboot open  http://$(hostname).local:8000  from a browser on the same network."
