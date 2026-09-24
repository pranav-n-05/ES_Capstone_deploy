"""
Web UI for the Raspberry Pi Marvin keyword spotter.

  python app.py            # then open http://<pi-hostname>.local:8000 from any browser on the same network
"""

import io
import os
import json
import time
import queue
import argparse
import threading
from collections import deque

import numpy as np
from scipy.io import wavfile
from flask import Flask, Response, jsonify, request, render_template, send_from_directory

from edge_kws import EdgeKWS, load_wav, log_mel_filterbank, normalize_level, rms, AUDIO_SR, WINDOW_SAMPLES
from audio_input import MicStream, list_input_devices, find_default_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

DEFAULT_SETTINGS = {
    "device": None,
    "channel": "auto",
    "gain_db": 20.0,
    "auto_level": True,
    "gate_rms": 40.0,
    "threshold": 0.0,
    "cooldown_s": 1.0,
}

MAX_GATE_RMS = 1000.0


def expected_label(file_name):
    """
    Ground truth encoded in the bundled sample names
    """

    name = file_name.lower()
    if name.startswith("marvin"):
        return 1
    if name.startswith("other") or name.startswith("silence"):
        return -1
    return None


class Engine:
    """
    Owns the detector, the microphone and the live detection loop
    """

    def __init__(self, threads):
        self.kws = EdgeKWS(num_threads=threads)
        self.kws_lock = threading.Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.mic = None
        self.mic_queue = None
        self.worker = None
        self.listening = False
        self.clients = []
        self.clients_lock = threading.Lock()
        self.detections = deque(maxlen=50)
        self.last_latency_ms = None

    # ---------- events ----------

    def broadcast(self, event):
        payload = json.dumps(event)
        with self.clients_lock:
            for q in self.clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def add_client(self):
        q = queue.Queue(maxsize=100)
        with self.clients_lock:
            self.clients.append(q)
        return q

    def remove_client(self, q):
        with self.clients_lock:
            if q in self.clients:
                self.clients.remove(q)

    # ---------- microphone ----------

    def _mic(self):
        if self.mic is None:
            self.mic = MicStream(device=self.settings["device"], channel=self.settings["channel"])
        return self.mic

    def reset_mic(self):
        was_listening = self.listening
        self.stop()
        if self.mic is not None:
            self.mic.stop()
        self.mic = None
        if was_listening:
            self.start()

    def start(self):
        if self.listening:
            return self.status()

        config = self._mic().start()
        self.mic_queue = self.mic.subscribe()
        self.listening = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.broadcast({"type": "state", "listening": True, "mic": config})

        return self.status()

    def stop(self):
        if not self.listening:
            return self.status()

        self.listening = False
        if self.worker is not None:
            self.worker.join(timeout=2)
        if self.mic is not None:
            self.mic.unsubscribe(self.mic_queue)
            self.mic.stop()
        self.broadcast({"type": "state", "listening": False})

        return self.status()

    def _loop(self):
        buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        last_detection = 0.0

        while self.listening:
            try:
                chunk = self.mic_queue.get(timeout=1)
            except queue.Empty:
                continue

            s = self.settings
            gain = 10 ** (s["gain_db"] / 20)
            buffer = np.concatenate([buffer, chunk * gain])[-WINDOW_SAMPLES:]
            level = rms(buffer)
            chunk_level = rms(chunk * gain)

            score, latency_ms = None, None
            window = normalize_level(buffer) if s["auto_level"] else buffer

            if level >= s["gate_rms"]:
                start = time.perf_counter()
                with self.kws_lock:
                    score = float(self.kws.decision(self.kws.embed(log_mel_filterbank(window)))[0])
                latency_ms = 1000 * (time.perf_counter() - start)
                self.last_latency_ms = latency_ms

            detected = False
            now = time.time()
            if score is not None and score >= s["threshold"] and now - last_detection >= s["cooldown_s"]:
                detected = True
                last_detection = now
                self.detections.appendleft({"time": now, "score": score, "latency_ms": latency_ms, "rms": level})

            # Compact visuals: waveform envelope and a coarse spectrogram
            envelope = np.abs(buffer).reshape(100, -1).max(axis=1)
            spectrogram = log_mel_filterbank(window)[::2]

            self.broadcast(
                {
                    "type": "frame",
                    "t": now,
                    "rms": level,
                    "chunk_rms": chunk_level,
                    "score": score,
                    "detected": detected,
                    "latency_ms": latency_ms,
                    "envelope": np.round(envelope, 1).tolist(),
                    "spectrogram": np.round(spectrogram, 2).tolist(),
                }
            )

    def calibrate(self, seconds=2.0):
        """
        Measures background noise and sets the silence gate just above it
        """

        audio = self._mic().record(seconds) * 10 ** (self.settings["gain_db"] / 20)
        windows = [rms(audio[i : i + 4000]) for i in range(0, len(audio) - 3999, 4000)]
        noise = float(np.percentile(windows, 90)) if windows else 0.0
        # Cap the gate so a noisy calibration cannot silence the detector
        self.settings["gate_rms"] = round(min(max(3.0 * noise, 5.0), MAX_GATE_RMS), 1)
        warning = None
        if 3.0 * noise > MAX_GATE_RMS:
            warning = "Background is very loud (noise RMS {:.0f}). Lower the gain or calibrate in silence.".format(noise)

        return {"noise_rms": noise, "gate_rms": self.settings["gate_rms"], "warning": warning}

    def record(self, seconds):
        audio = self._mic().record(seconds)
        return audio * 10 ** (self.settings["gain_db"] / 20)

    # ---------- offline analysis ----------

    def analyze(self, wave):
        s = self.settings
        with self.kws_lock:
            return self.kws.analyze(
                wave, normalize=s["auto_level"], threshold=s["threshold"], gate_rms=s["gate_rms"]
            )

    def status(self):
        return {
            "listening": self.listening,
            "mic": self.mic.config if self.mic is not None else None,
            "settings": self.settings,
            "last_latency_ms": self.last_latency_ms,
            "detections": list(self.detections),
        }


def create_app(threads=4):
    app = Flask(__name__)
    engine = Engine(threads)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/favicon.ico")
    def favicon():
        return "", 204

    @app.route("/api/status")
    def status():
        return jsonify(engine.status())

    @app.route("/api/devices")
    def devices():
        try:
            return jsonify({"devices": list_input_devices(), "suggested": find_default_device()})
        except Exception as error:
            return jsonify({"devices": [], "suggested": None, "error": str(error)})

    @app.route("/api/settings", methods=["POST"])
    def settings():
        data = request.get_json(force=True)
        mic_changed = False

        for key in ("gain_db", "gate_rms", "threshold", "cooldown_s"):
            if key in data:
                engine.settings[key] = float(data[key])
        if "auto_level" in data:
            engine.settings["auto_level"] = bool(data["auto_level"])
        if "channel" in data and data["channel"] != engine.settings["channel"]:
            engine.settings["channel"] = data["channel"]
            mic_changed = True
        if "device" in data:
            device = None if data["device"] in (None, "", "auto") else int(data["device"])
            if device != engine.settings["device"]:
                engine.settings["device"] = device
                mic_changed = True

        if mic_changed:
            engine.reset_mic()

        return jsonify(engine.status())

    @app.route("/api/listen/start", methods=["POST"])
    def listen_start():
        try:
            return jsonify(engine.start())
        except Exception as error:
            return jsonify({"error": "Could not open microphone: {}".format(error)}), 500

    @app.route("/api/listen/stop", methods=["POST"])
    def listen_stop():
        return jsonify(engine.stop())

    @app.route("/api/calibrate", methods=["POST"])
    def calibrate():
        try:
            return jsonify(engine.calibrate())
        except Exception as error:
            return jsonify({"error": str(error)}), 500

    @app.route("/api/events")
    def events():
        q = engine.add_client()

        def stream():
            try:
                yield "data: {}\n\n".format(json.dumps({"type": "hello", **engine.status()}))
                while True:
                    try:
                        yield "data: {}\n\n".format(q.get(timeout=15))
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                engine.remove_client(q)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.route("/api/samples")
    def samples():
        files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav"))
        return jsonify({"samples": [{"name": f, "expected": expected_label(f)} for f in files]})

    @app.route("/samples/<path:name>")
    def sample_file(name):
        return send_from_directory(SAMPLES_DIR, name, mimetype="audio/wav")

    @app.route("/recordings/<path:name>")
    def recording_file(name):
        return send_from_directory(RECORDINGS_DIR, name, mimetype="audio/wav")

    @app.route("/api/analyze/sample/<path:name>", methods=["POST"])
    def analyze_sample(name):
        path = os.path.join(SAMPLES_DIR, os.path.basename(name))
        if not os.path.isfile(path):
            return jsonify({"error": "Unknown sample"}), 404
        result = engine.analyze(load_wav(path))
        result.update({"name": os.path.basename(path), "url": "/samples/" + os.path.basename(path)})
        return jsonify(result)

    @app.route("/api/analyze/upload", methods=["POST"])
    def analyze_upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename.lower().endswith(".wav"):
            return jsonify({"error": "Please upload a .wav file"}), 400
        try:
            wave = load_wav(io.BytesIO(uploaded.read()))
        except Exception as error:
            return jsonify({"error": "Could not read wav: {}".format(error)}), 400

        name = "upload_{}.wav".format(time.strftime("%Y%m%d_%H%M%S"))
        wavfile.write(os.path.join(RECORDINGS_DIR, name), AUDIO_SR, np.clip(wave, -32768, 32767).astype(np.int16))
        result = engine.analyze(wave)
        result.update({"name": uploaded.filename, "url": "/recordings/" + name})
        return jsonify(result)

    @app.route("/api/record", methods=["POST"])
    def record():
        seconds = min(max(float(request.args.get("seconds", 3)), 1.0), 15.0)
        try:
            wave = engine.record(seconds)
        except Exception as error:
            return jsonify({"error": "Could not record: {}".format(error)}), 500

        name = "rec_{}.wav".format(time.strftime("%Y%m%d_%H%M%S"))
        wavfile.write(os.path.join(RECORDINGS_DIR, name), AUDIO_SR, np.clip(wave, -32768, 32767).astype(np.int16))
        result = engine.analyze(wave)
        result.update({"name": name, "url": "/recordings/" + name, "rms": rms(wave)})
        return jsonify(result)

    @app.route("/api/selftest", methods=["POST"])
    def selftest():
        rows = []
        for f in sorted(os.listdir(SAMPLES_DIR)):
            expected = expected_label(f)
            if expected is None:
                continue
            result = engine.analyze(load_wav(os.path.join(SAMPLES_DIR, f)))
            got = 1 if result["detected"] else -1
            rows.append(
                {
                    "name": f,
                    "expected": expected,
                    "got": got,
                    "pass": got == expected,
                    "max_score": result["max_score"],
                    "ms_per_window": result["ms_per_window"],
                }
            )

        tp = sum(r["expected"] == 1 and r["got"] == 1 for r in rows)
        fp = sum(r["expected"] == -1 and r["got"] == 1 for r in rows)
        fn = sum(r["expected"] == 1 and r["got"] == -1 for r in rows)

        return jsonify(
            {
                "rows": rows,
                "passed": sum(r["pass"] for r in rows),
                "total": len(rows),
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "avg_ms_per_window": float(np.mean([r["ms_per_window"] for r in rows])) if rows else None,
            }
        )

    app.engine = engine
    return app


def main():
    parser = argparse.ArgumentParser(description="Marvin keyword spotting web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=4, help="TFLite interpreter threads")
    args = parser.parse_args()

    app = create_app(args.threads)
    print("Open http://<this-pi>.local:{} in a browser".format(args.port))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
