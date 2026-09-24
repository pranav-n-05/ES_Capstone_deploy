# Marvin keyword spotting on Raspberry Pi 4 Model B + INMP441

Self-contained edge deployment: int8 TFLite CNN + numpy PCA/one-class SVM, INMP441 I2S microphone, browser UI.
Nothing outside this folder is needed on the Pi, and TensorFlow / scikit-learn are **not** installed there.

## Files

| File | Purpose |
| :-- | :-- |
| `app.py` | Web UI server (Flask). Open `http://<pi>.local:8000` from any browser on the same network |
| `templates/index.html` | The UI (live detection, settings, manual tests). No internet needed |
| `edge_kws.py` | Detector: log-Mel features, int8 TFLite CNN, numpy PCA + one-class SVM |
| `audio_input.py` | INMP441 capture: 32-bit I2S stereo → channel pick → DC removal → 16 kHz |
| `kws_cli.py` | Terminal fallback (live, `--wav`, `--selftest`, `--list-devices`) |
| `models/marvin_kws_int8.tflite` | Quantised CNN feature extractor (941 KB) |
| `models/marvin_kws_svm.npz` | PCA mean/components + SVM support vectors, weights, intercept, gamma |
| `samples/*.wav` | 20 unseen test clips (`marvin_*`, `other_*`, `silence_*`) + a 14 s demo with 3 “Marvin”s |
| `requirements.txt` | Pi Python packages |
| `setup_pi.sh` | One-time setup: packages, I2S overlay, venv, autostart service |
| `kws-web.service` | systemd unit template that starts the UI at boot |

## 1. Hardware

- Raspberry Pi 4 Model B, official 5.1 V / 3 A USB-C supply, microSD ≥ 16 GB with **Raspberry Pi OS 64-bit** (Bookworm or newer)
- INMP441 I2S MEMS microphone module + 6 female–female jumper wires

The INMP441 is an **I2S** (digital audio) microphone, not I2C. Wire it with the Pi **powered off**:

| INMP441 pin | Raspberry Pi 4 pin | GPIO / function |
| :-- | :-- | :-- |
| VDD | Pin 1 | 3.3 V (**not 5 V**) |
| GND | Pin 6 | Ground |
| L/R | Pin 9 | Ground → left channel |
| SCK | Pin 12 | GPIO 18 · PCM_CLK (bit clock) |
| WS | Pin 35 | GPIO 19 · PCM_FS (word select) |
| SD | Pin 38 | GPIO 20 · PCM_DIN (data in) |

```
Pi 4 header (USB ports at the bottom)      INMP441
 pin 1  3V3  ●○  pin 2                      VDD ── pin 1
 ...                                         GND ── pin 6
 pin 9  GND  ●○                              L/R ── pin 9
 pin 11      ○●  pin 12 GPIO18  ─────────── SCK
 ...                                         WS  ── pin 35
 pin 35 GPIO19 ●○                            SD  ── pin 38
 pin 37      ○●  pin 38 GPIO20
```

The sound hole is on the underside of the INMP441 board; do not cover it.

## 2. Install (once)

```shell
git clone https://github.com/pranav-n-05/ES_Capstone_deploy.git ~/kws
cd ~/kws/deploy/pi
bash setup_pi.sh        # ~5 min: apt packages, I2S overlay, Python venv, autostart service
sudo reboot
```

`setup_pi.sh` adds these two lines to `/boot/firmware/config.txt` (a backup is saved as `config.txt.bak.kws`):

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

## 3. Check the microphone (after reboot)

```shell
arecord -l                       # expect a card named "sndrpigooglevoi" (card number N)
arecord -D plughw:N,0 -c 2 -r 48000 -f S32_LE -d 5 test.wav   # speak for 5 s
aplay test.wav                   # needs speakers/headphones; otherwise open it in the UI (Upload WAV)
cd ~/kws/deploy/pi && .venv/bin/python kws_cli.py --list-devices   # the I2S mic is marked
```

## 4. Run the UI

The service starts it automatically at boot. From a laptop or phone on the **same Wi-Fi / hotspot** open:

```
http://kwspi.local:8000        (use your Pi's hostname, or its IP address)
```

Manual start/stop and logs:

```shell
sudo systemctl status kws-web        # running?
sudo systemctl restart kws-web
journalctl -u kws-web -f             # live logs
# or run in the foreground instead:
sudo systemctl stop kws-web && cd ~/kws/deploy/pi && .venv/bin/python app.py
```

First time with the mic: press **Start listening**, stay quiet and press **Calibrate noise (2 s)**, then speak.
The level bar should turn teal (above the yellow gate mark) when you talk and drop below it in silence.

## 5. Manual test plan for the demo

Run with default settings (threshold 0.00, auto level on, gain +20 dB) unless the step says otherwise.
Fill in the last column during rehearsal.

| # | Test | Steps | Expected result | Result |
| :-: | :-- | :-- | :-- | :-- |
| T1 | Model loads | Open the UI | Header shows `int8 TFLite · 0.94 MB` and `connected` | |
| T2 | Self-test | Manual tests → Self-test → Run | 19/20 correct, precision 1.00; `marvin_04` fails (score −0.15) | |
| T3 | Threshold trade-off | Set threshold to −0.20, run self-test again | 20/20 correct; set threshold back to 0.00 | |
| T4 | Offline recording | Test clips → `long_demo_3x_marvin.wav` → Analyse | 3 detections at ≈2 s, 6.5 s, 11 s; plot shows 3 green regions | |
| T5 | Negative clip | Analyse `other_sheila.wav` (similar sounding) | “No Marvin”, score ≈ −2.9 | |
| T6 | Mic works | Start listening, speak | Level bar moves, spectrogram and waveform update 4×/s | |
| T7 | Live keyword | Say “Marvin” 5 times, 3 s apart, ~30–50 cm from the mic | Green MARVIN! flash and a log entry for most (≥ 4/5) | |
| T8 | Live rejection | Say “yes”, “house”, “Sheila”, “go”, “seven” | No detections (a rare false alarm is possible) | |
| T9 | Silence | Stay quiet for 30 s | Indicator shows “quiet”, no detections | |
| T10 | Latency | Read the `latency` pill while listening | A few ms per window (compare with 250 ms hop) | |
| T11 | Record & analyse | Record tab → 4 s → say “Marvin” once | Detection shown on the timeline; recording plays back | |
| T12 | Upload | Upload WAV → any `.wav` recorded on a phone/laptop | Analysis result with timeline | |

## 6. Troubleshooting

| Symptom | Fix |
| :-- | :-- |
| `arecord -l` shows no `sndrpigooglevoi` | Check `/boot/firmware/config.txt` has both lines, reboot, re-check wiring (SCK 12, WS 35, SD 38) |
| Level bar never moves / RMS ≈ 0 | SD or WS wire wrong; VDD must be 3.3 V; try Channel = left/right |
| Level always below the gate | Increase Gain, or press Calibrate noise in silence |
| Many false alarms in a noisy room | Raise threshold to 0.2–0.3, calibrate the gate, speak closer to the mic |
| `Could not open microphone` | Stop other programs using the mic (`arecord`), pick the I2S device in Settings |
| Page does not open | Same network? Try the Pi IP (`hostname -I`); check `systemctl status kws-web` |
| No live mic at all | Use T2–T5 and T12: the model still runs fully on the Pi from files |
