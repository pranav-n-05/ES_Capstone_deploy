"""
Lightweight Marvin detector for the Raspberry Pi.
Needs only numpy, scipy, python_speech_features and a TFLite runtime - no TensorFlow or scikit-learn.
"""

import os
import time
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from python_speech_features import logfbank

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

AUDIO_SR = 16000
WINDOW_SAMPLES = 16000
HOP_SAMPLES = 4000
PARSE_PARAMS = (0.025, 0.01, 40)

# Median RMS of the training clips (int16 units); windows are scaled to this level
TARGET_RMS = 1500.0

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def log_mel_filterbank(wave):
    """
    Computes the log Mel filterbanks for a 1 s window, same settings as training
    :param wave: Audio as an array (int16 scale)
    :return: Filterbanks (99, 40)
    """

    wave = np.asarray(wave, dtype=np.float32)

    # Pad with noise if audio is short
    if len(wave) < WINDOW_SAMPLES:
        wave = np.append(wave, np.random.normal(0, 5, WINDOW_SAMPLES - len(wave))).astype(np.float32)

    fbank = logfbank(
        wave[:WINDOW_SAMPLES],
        samplerate=AUDIO_SR,
        winlen=PARSE_PARAMS[0],
        winstep=PARSE_PARAMS[1],
        highfreq=AUDIO_SR / 2,
        nfilt=PARSE_PARAMS[2],
    )

    return np.array(fbank, dtype=np.float32)


def rms(wave):
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if len(wave) else 0.0


def normalize_level(wave, target_rms=TARGET_RMS):
    """
    Scales a window to the typical training loudness
    :param wave: Audio window
    :param target_rms: Desired RMS
    :return: Scaled audio
    """

    return wave * (target_rms / max(rms(wave), 1.0))


def load_wav(path):
    """
    Reads a wav file as 16 kHz mono in int16 scale
    :param path: Wav file path (or file-like object)
    :return: Audio array (float32)
    """

    sr, wave = wavfile.read(path)

    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    # Bring every sample format to int16 scale
    if wave.dtype == np.int32:
        wave = wave / 65536.0
    elif wave.dtype == np.uint8:
        wave = (wave.astype(np.float32) - 128) * 256
    elif np.issubdtype(wave.dtype, np.floating):
        wave = wave * 32767.0

    if sr != AUDIO_SR:
        factor = gcd(sr, AUDIO_SR)
        wave = resample_poly(wave, AUDIO_SR // factor, sr // factor)

    return np.asarray(wave, dtype=np.float32)


class EdgeKWS:
    """
    int8 TFLite feature extractor + numpy PCA + numpy one-class SVM
    """

    def __init__(self, model_dir=MODEL_DIR, num_threads=4):
        self.interpreter = Interpreter(
            model_path=os.path.join(model_dir, "marvin_kws_int8.tflite"), num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        params = np.load(os.path.join(model_dir, "marvin_kws_svm.npz"))
        self.pca_mean = params["pca_mean"]
        self.pca_components = params["pca_components"]
        self.support_vectors = params["support_vectors"]
        self.dual_coef = params["dual_coef"]
        self.intercept = float(params["intercept"])
        self.gamma = float(params["gamma"])

    def embed(self, fbank):
        """
        Obtain the 256-d embedding for one window
        :param fbank: Filterbanks (99, 40)
        :return: Embedding (256,)
        """

        self.interpreter.set_tensor(self.input_index, fbank[np.newaxis].astype(np.float32))
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_index)[0]

    def decision(self, embeddings):
        """
        One-class SVM decision function (RBF kernel) on PCA-reduced embeddings
        :param embeddings: Embeddings (n, 256)
        :return: Decision values (n,), >= 0 means Marvin
        """

        x = (np.atleast_2d(embeddings) - self.pca_mean) @ self.pca_components.T
        sq_dist = ((x[:, np.newaxis, :] - self.support_vectors[np.newaxis]) ** 2).sum(axis=-1)

        return np.exp(-self.gamma * sq_dist) @ self.dual_coef + self.intercept

    def score(self, wave, normalize=True):
        """
        Decision score for one 1 s window
        :param wave: Audio window (int16 scale)
        :param normalize: Whether to scale the window to training loudness
        :return: Score (>= 0 means Marvin)
        """

        if normalize:
            wave = normalize_level(wave)

        return float(self.decision(self.embed(log_mel_filterbank(wave)))[0])

    def predict(self, wave, normalize=True, threshold=0.0):
        return 1 if self.score(wave, normalize) >= threshold else -1

    def analyze(self, wave, normalize=True, threshold=0.0, gate_rms=0.0):
        """
        Slides a 1 s window (0.25 s hop) over a recording
        :param wave: Audio (int16 scale, 16 kHz)
        :param normalize: Whether to scale windows to training loudness
        :param threshold: Decision threshold
        :param gate_rms: Windows quieter than this are treated as silence
        :return: Dictionary with per-window scores and merged detections
        """

        if len(wave) < WINDOW_SAMPLES:
            wave = np.append(wave, np.zeros(WINDOW_SAMPLES - len(wave), dtype=np.float32))

        windows = []
        start_time = time.perf_counter()

        for start in range(0, len(wave) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
            window = wave[start : start + WINDOW_SAMPLES]
            level = rms(window)
            score = self.score(window, normalize) if level >= gate_rms else None
            windows.append({"t": start / AUDIO_SR, "rms": level, "score": score})

        elapsed_ms = 1000 * (time.perf_counter() - start_time)

        # Merge overlapping positive windows into single events
        events = []
        for w in windows:
            if w["score"] is None or w["score"] < threshold:
                continue
            if events and w["t"] - events[-1]["end"] <= 0.5 + 1e-6:
                events[-1]["end"] = w["t"]
                events[-1]["score"] = max(events[-1]["score"], w["score"])
            else:
                events.append({"start": w["t"], "end": w["t"], "score": w["score"]})

        for event in events:
            event["end"] += WINDOW_SAMPLES / AUDIO_SR

        return {
            "duration": len(wave) / AUDIO_SR,
            "windows": windows,
            "events": events,
            "detected": len(events) > 0,
            "max_score": max((w["score"] for w in windows if w["score"] is not None), default=None),
            "ms_per_window": elapsed_ms / max(len(windows), 1),
        }
