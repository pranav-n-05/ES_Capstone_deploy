"""
Edge (Raspberry Pi) Marvin keyword spotting demo.

  python stream_audio_pi.py                  # live, default microphone
  python stream_audio_pi.py --device 2       # live, specific input device
  python stream_audio_pi.py --list-devices   # show input devices
  python stream_audio_pi.py --wav clip.wav   # offline, slide over a recording
"""

import sys
import time
import argparse
import numpy as np
from queue import Queue
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly

from edge_kws import EdgeKWS, log_mel_filterbank, AUDIO_SR

WINDOW_SAMPLES = AUDIO_SR
HOP_SAMPLES = int(0.25 * AUDIO_SR)
CHUNK_SAMPLES = int(0.25 * AUDIO_SR)
SILENCE_THRESHOLD = 100


def run_wav(kws, wav_path):
    """
    Slide a 1 s window over a recording and report Marvin detections
    :param kws: EdgeKWS detector
    :param wav_path: Path to wav file
    :return: None
    """

    sr, wave = wavfile.read(wav_path)

    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    if sr != AUDIO_SR:
        factor = gcd(sr, AUDIO_SR)
        wave = resample_poly(wave, AUDIO_SR // factor, sr // factor)

    wave = wave.astype(np.float32)
    starts = range(0, max(len(wave) - WINDOW_SAMPLES, 0) + 1, HOP_SAMPLES)
    print("{}: {:.1f} s, {} windows".format(wav_path, len(wave) / AUDIO_SR, len(starts)))

    detections = []
    start_time = time.perf_counter()

    for start in starts:
        if kws.predict(log_mel_filterbank(wave[start : start + WINDOW_SAMPLES])) == 1:
            detections.append(start / AUDIO_SR)

    elapsed = time.perf_counter() - start_time

    # Merge overlapping windows into single events
    events = []
    for t in detections:
        if events and t - events[-1][1] <= 0.5:
            events[-1][1] = t
        else:
            events.append([t, t])

    for begin, end in events:
        print("Marvin! at {:.2f}-{:.2f} s".format(begin, end + 1))

    print("{} detection(s). {:.2f} ms per window".format(len(events), 1000 * elapsed / max(len(starts), 1)))


def run_mic(kws, device):
    """
    Stream from the microphone and report Marvin detections
    :param kws: EdgeKWS detector
    :param device: PyAudio input device index (None for default)
    :return: None
    """

    import pyaudio

    queue = Queue()
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.int16)

    def callback(in_data, frame_count, time_info, status):
        nonlocal buffer
        buffer = np.append(buffer, np.frombuffer(in_data, dtype=np.int16))[-WINDOW_SAMPLES:]
        queue.put(buffer.copy())
        return in_data, pyaudio.paContinue

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=AUDIO_SR,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
        input_device_index=device,
        stream_callback=callback,
    )

    print("Listening... say 'Marvin' (Ctrl+C to stop)")
    stream.start_stream()
    cooldown = 0

    try:
        while True:
            window = queue.get()
            start = time.perf_counter()
            pred = kws.predict(log_mel_filterbank(window))
            elapsed_ms = 1000 * (time.perf_counter() - start)

            if pred == 1 and cooldown == 0:
                print("\nMarvin!  ({:.1f} ms)".format(elapsed_ms), flush=True)
                cooldown = 3
            else:
                cooldown = max(cooldown - 1, 0)
                marker = "." if np.abs(window[-CHUNK_SAMPLES:]).mean() < SILENCE_THRESHOLD else "-"
                print(marker, end="", flush=True)

    except KeyboardInterrupt:
        print()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def list_devices():
    import pyaudio

    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print("{}: {} ({} ch, {:.0f} Hz)".format(i, info["name"], info["maxInputChannels"], info["defaultSampleRate"]))
    audio.terminate()


def main():
    parser = argparse.ArgumentParser(description="Edge Marvin keyword spotting")
    parser.add_argument("--wav", help="Run on a wav file instead of the microphone")
    parser.add_argument("--device", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    parser.add_argument("--models", default="../models", help="Directory with the edge model files")
    parser.add_argument("--threads", type=int, default=4, help="TFLite interpreter threads")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    kws = EdgeKWS(args.models, num_threads=args.threads)

    if args.wav:
        run_wav(kws, args.wav)
    else:
        run_mic(kws, args.device)


if __name__ == "__main__":
    sys.exit(main())
