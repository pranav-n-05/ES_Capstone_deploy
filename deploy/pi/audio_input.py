"""
Microphone capture for the INMP441 I2S MEMS microphone (also works with USB mics).

The INMP441 delivers 24-bit samples in 32-bit stereo I2S frames. This module picks a working
ALSA configuration, selects the active channel (the L/R pin decides left or right), removes DC,
resamples to 16 kHz and emits 0.25 s chunks in int16 scale - the format the model was trained on.
"""

import queue
import threading

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

import sounddevice as sd

OUT_SR = 16000
CHUNK_SECONDS = 0.25

# Name fragments of I2S capture cards created by the common INMP441 overlays
I2S_NAME_HINTS = ("voicehat", "googlevoi", "i2s", "inmp441", "adau7002", "snd_rpi", "sndrpi")


def list_input_devices():
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                {
                    "index": index,
                    "name": info["name"],
                    "channels": info["max_input_channels"],
                    "default_samplerate": info["default_samplerate"],
                }
            )
    return devices


def find_default_device():
    """
    Prefer an I2S microphone card, then the system default input
    :return: Device index or None
    """

    for device in list_input_devices():
        if any(hint in device["name"].lower() for hint in I2S_NAME_HINTS):
            return device["index"]

    return None


class _Decimator:
    """
    Stateful low-pass + integer decimation, so 0.25 s blocks join without clicks
    """

    def __init__(self, factor):
        self.factor = factor
        self.taps = firwin(8 * factor * 4 + 1, 0.9 / factor)
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0

    def __call__(self, x):
        y, self.zi = lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.phase :: self.factor]
        self.phase = (self.phase - len(x)) % self.factor
        return out


class MicStream:
    """
    Opens the microphone and delivers 16 kHz, int16-scale float32 chunks to subscribers
    """

    def __init__(self, device=None, channel="auto"):
        self.device = find_default_device() if device is None else device
        self.channel = channel
        self.stream = None
        self.config = None
        self.subscribers = []
        self.lock = threading.Lock()
        self._energy = None
        self._dc = 0.0
        self._resample = None

    def _probe(self):
        """
        Finds a (rate, channels, dtype) the device accepts, preferring no resampling
        """

        info = sd.query_devices(self.device, "input")
        max_channels = int(info["max_input_channels"])
        channel_options = [c for c in (2, 1) if c <= max_channels] or [max_channels]

        for rate in (16000, 48000, 32000, 44100):
            for channels in channel_options:
                for dtype in ("int32", "int16", "float32"):
                    try:
                        sd.check_input_settings(device=self.device, samplerate=rate, channels=channels, dtype=dtype)
                        return {"rate": rate, "channels": channels, "dtype": dtype, "name": info["name"]}
                    except Exception:
                        continue

        raise RuntimeError("No supported input configuration for device '{}'".format(info["name"]))

    def start(self):
        if self.stream is not None:
            return self.config

        self.config = self._probe()
        rate = self.config["rate"]

        if rate == OUT_SR:
            self._resample = None
        elif rate % OUT_SR == 0:
            self._resample = _Decimator(rate // OUT_SR)
        else:
            self._resample = lambda x: resample_poly(x, 160, 441) if rate == 44100 else resample_poly(x, OUT_SR, rate)

        self._energy = None
        self._dc = 0.0
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=rate,
            channels=self.config["channels"],
            dtype=self.config["dtype"],
            blocksize=int(rate * CHUNK_SECONDS),
            callback=self._callback,
        )
        self.stream.start()

        return self.config

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    @property
    def running(self):
        return self.stream is not None

    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _select_channel(self, block):
        if block.shape[1] == 1:
            return block[:, 0]
        if self.channel == "left":
            return block[:, 0]
        if self.channel == "right":
            return block[:, 1]

        # Auto: the unconnected I2S slot carries (near) silence, pick the louder one
        energy = np.mean(np.square(block, dtype=np.float64), axis=0)
        self._energy = energy if self._energy is None else 0.9 * self._energy + 0.1 * energy
        return block[:, int(np.argmax(self._energy))]

    def _callback(self, indata, frames, time_info, status):
        block = indata.astype(np.float64)

        # Convert to int16 scale
        dtype = self.config["dtype"]
        if dtype == "int32":
            block /= 65536.0
        elif dtype == "float32":
            block *= 32767.0

        mono = self._select_channel(block)

        # Remove DC offset with a slow running mean
        self._dc = 0.995 * self._dc + 0.005 * float(np.mean(mono))
        mono = mono - self._dc

        if self._resample is not None:
            mono = self._resample(mono)

        chunk = mono.astype(np.float32)

        with self.lock:
            for q in self.subscribers:
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass

    def record(self, seconds):
        """
        Records a fixed duration (starts the stream temporarily if needed)
        :param seconds: Duration
        :return: 16 kHz audio (int16 scale)
        """

        was_running = self.running
        self.start()
        q = self.subscribe()
        chunks, collected = [], 0

        try:
            while collected < seconds * OUT_SR:
                chunk = q.get(timeout=5)
                chunks.append(chunk)
                collected += len(chunk)
        finally:
            self.unsubscribe(q)
            if not was_running:
                self.stop()

        return np.concatenate(chunks)[: int(seconds * OUT_SR)]
