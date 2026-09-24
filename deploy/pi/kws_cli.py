"""
Terminal version of the demo (fallback if the browser UI is unavailable).

  python kws_cli.py --list-devices          # show input devices
  python kws_cli.py                         # live, INMP441 / default microphone
  python kws_cli.py --wav samples/long_demo_3x_marvin.wav
  python kws_cli.py --selftest              # run all bundled test clips
"""

import os
import time
import argparse

import numpy as np

from edge_kws import EdgeKWS, load_wav, rms, WINDOW_SAMPLES

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def run_wav(kws, path, args):
    result = kws.analyze(load_wav(path), normalize=not args.no_auto_level, threshold=args.threshold, gate_rms=args.gate)
    print("{}: {:.1f} s, {} windows, {:.2f} ms per window".format(path, result["duration"], len(result["windows"]), result["ms_per_window"]))
    for event in result["events"]:
        print("  Marvin! at {:.2f}-{:.2f} s (score {:.2f})".format(event["start"], event["end"], event["score"]))
    if not result["events"]:
        print("  no Marvin (max score {})".format("-" if result["max_score"] is None else "{:.2f}".format(result["max_score"])))


def run_selftest(kws, args):
    passed = total = 0
    for name in sorted(os.listdir(SAMPLES_DIR)):
        if not name.startswith(("marvin", "other", "silence")):
            continue
        result = kws.analyze(load_wav(os.path.join(SAMPLES_DIR, name)), threshold=args.threshold, gate_rms=args.gate)
        expected = name.startswith("marvin")
        ok = result["detected"] == expected
        passed += ok
        total += 1
        score = "-" if result["max_score"] is None else "{:+.2f}".format(result["max_score"])
        print("{:<22} expected={:<6} got={:<6} score={:>6}  {}".format(
            name, "marvin" if expected else "other", "marvin" if result["detected"] else "other", score, "PASS" if ok else "FAIL"))
    print("{}/{} correct".format(passed, total))


def run_mic(kws, args):
    from audio_input import MicStream

    mic = MicStream(device=args.device, channel=args.channel)
    config = mic.start()
    print("Mic: {name} · {rate} Hz · {channels} ch · {dtype}".format(**config))
    print("Listening... say 'Marvin' (Ctrl+C to stop)")

    q = mic.subscribe()
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    gain = 10 ** (args.gain / 20)
    last = 0.0

    try:
        while True:
            buffer = np.concatenate([buffer, q.get() * gain])[-WINDOW_SAMPLES:]
            level = rms(buffer)
            if level < args.gate:
                print(".", end="", flush=True)
                continue

            start = time.perf_counter()
            score = kws.score(buffer, normalize=not args.no_auto_level)
            ms = 1000 * (time.perf_counter() - start)

            if score >= args.threshold and time.time() - last > 1.0:
                last = time.time()
                print("\nMarvin!  score {:.2f} · {:.1f} ms · rms {:.0f}".format(score, ms, level), flush=True)
            else:
                print("-", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        mic.stop()


def main():
    parser = argparse.ArgumentParser(description="Marvin keyword spotting (terminal)")
    parser.add_argument("--wav", help="Analyse a wav file")
    parser.add_argument("--selftest", action="store_true", help="Run all bundled test clips")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None, help="Input device index (default: I2S mic if found)")
    parser.add_argument("--channel", default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--gain", type=float, default=20.0, help="Gain in dB before the silence gate")
    parser.add_argument("--gate", type=float, default=40.0, help="Silence gate (RMS, int16 scale)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Decision threshold")
    parser.add_argument("--no-auto-level", action="store_true", help="Disable per-window level normalisation")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.list_devices:
        from audio_input import list_input_devices, find_default_device

        suggested = find_default_device()
        for d in list_input_devices():
            mark = "  <- I2S mic" if d["index"] == suggested else ""
            print("{index}: {name} ({channels} ch, {default_samplerate:.0f} Hz){mark}".format(mark=mark, **d))
        return

    kws = EdgeKWS(num_threads=args.threads)

    if args.selftest:
        run_selftest(kws, args)
    elif args.wav:
        run_wav(kws, args.wav, args)
    else:
        run_mic(kws, args)


if __name__ == "__main__":
    main()
