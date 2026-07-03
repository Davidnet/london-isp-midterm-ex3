#!/usr/bin/env python3
"""
Exercise 3.1 — Ultrasonic AM secret-code detector & decoder
===========================================================

The creator of a set of suspicious WAV files hid a four-number secret code by
*amplitude-modulating* it onto an ultrasonic carrier and mixing that with an
ordinary (audible) cover sound. Because the code lives above ~18 kHz it is
inaudible on normal playback.

This application:

  1. DETECTS which file carries suspicious data, by measuring how much spectral
     energy sits in the ultrasonic band (>= 18 kHz) relative to the whole signal.
  2. Finds the AM carrier frequency (the dominant ultrasonic spectral peak).
  3. DEMODULATES the carrier back down to the audible range (envelope detection,
     since the carrier is transmitted -> classic AM), recovering the code.
  4. Saves the recovered code as a WAV and PLAYS it out loud.

Usage:
    uv run ultrasonic_decoder.py                 # analyse all, decode + play winner
    uv run ultrasonic_decoder.py --no-play       # analyse + save, don't play
    uv run ultrasonic_decoder.py --file Ex3_sound4.wav   # force a specific file
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert

ULTRASONIC_HZ = 18000.0     # everything at/above this is "ultrasonic" for our files
CODE_BW_HZ = 3500.0         # audible bandwidth to keep for the recovered code

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "assets" / "Ex3_files"
OUT_DIR = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV as mono float64 in [-1, 1]."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # ignore stray non-data chunks
        sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    return sr, x


def save_wav(path: Path, sr: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(x)) + 1e-12
    wavfile.write(path, sr, (x / peak * 0.9 * 32767).astype(np.int16))


# --------------------------------------------------------------------------- #
# Step 1 & 2 — detection and carrier estimation
# --------------------------------------------------------------------------- #
def analyse(sr: int, x: np.ndarray) -> dict:
    """Measure ultrasonic energy fraction and locate the carrier peak."""
    win = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(win))
    freq = np.fft.rfftfreq(len(x), 1 / sr)

    total = mag.sum() + 1e-12
    ultra_mask = freq >= ULTRASONIC_HZ

    # Ignore the single Nyquist bin: on 44.1 kHz files it collects filter
    # edge/aliasing junk that is not a real hidden carrier.
    ultra_mask &= freq < (sr / 2 - 1.0)

    ultra_frac = mag[ultra_mask].sum() / total
    if ultra_mask.any():
        carrier = float(freq[ultra_mask][np.argmax(mag[ultra_mask])])
        carrier_mag = float(mag[ultra_mask].max())
    else:
        carrier, carrier_mag = 0.0, 0.0

    return {
        "sr": sr,
        "duration": len(x) / sr,
        "nyquist": sr / 2,
        "ultra_frac": ultra_frac,
        "carrier": carrier,
        "carrier_mag": carrier_mag,
    }


# --------------------------------------------------------------------------- #
# Step 3 — AM demodulation (ultrasonic -> audible)
# --------------------------------------------------------------------------- #
def demodulate(sr: int, x: np.ndarray, carrier: float,
               half_bw: float = 2500.0) -> np.ndarray:
    """
    Recover the modulating code from an AM ultrasonic carrier.

    Band-pass around the carrier to isolate the AM signal, take the envelope
    (magnitude of the analytic signal) to undo the modulation, remove DC, and
    low-pass to keep only the audible code band.
    """
    lo = max(carrier - half_bw, 100.0)
    hi = min(carrier + half_bw, sr / 2 - 100.0)

    bp = butter(6, [lo, hi], btype="band", fs=sr, output="sos")
    band = sosfiltfilt(bp, x)

    envelope = np.abs(hilbert(band))          # AM envelope = the code
    envelope -= np.mean(envelope)             # drop the DC term

    lp = butter(6, CODE_BW_HZ, btype="low", fs=sr, output="sos")
    code = sosfiltfilt(lp, envelope)

    code /= np.max(np.abs(code)) + 1e-12
    return code


# --------------------------------------------------------------------------- #
# Step 4 — playback
# --------------------------------------------------------------------------- #
def play(sr: int, x: np.ndarray) -> None:
    try:
        import sounddevice as sd
    except Exception as e:                    # pragma: no cover
        print(f"  [playback unavailable: {e}]")
        return
    try:
        print("  Playing the secret code ...")
        sd.play(x, sr)
        sd.wait()
    except Exception as e:                    # pragma: no cover
        print(f"  [could not play audio: {e}] — use the saved WAV instead.")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="folder with the Ex3_sound*.wav files")
    ap.add_argument("--file", type=str, default=None,
                    help="force decoding of a specific file name")
    ap.add_argument("--no-play", action="store_true", help="do not play audio")
    args = ap.parse_args()

    files = sorted(args.dir.glob("Ex3_sound[1-4].wav"))
    if not files:
        print(f"No Ex3_sound*.wav files found in {args.dir}", file=sys.stderr)
        return 1

    print("=" * 68)
    print("Step 1-2 : Scanning for ultrasonic (amplitude-modulated) content")
    print("=" * 68)
    print(f"{'file':<18}{'sr':>7}{'ultra≥18kHz':>13}{'carrier':>11}")
    print("-" * 68)

    reports = {}
    for f in files:
        sr, x = load_wav(f)
        r = analyse(sr, x)
        reports[f] = (sr, x, r)
        print(f"{f.name:<18}{r['sr']:>7}{r['ultra_frac']*100:>11.3f}%"
              f"{r['carrier']:>10.0f}Hz")

    # The suspicious file = largest ultrasonic energy fraction.
    if args.file:
        winner = args.dir / args.file
        if winner not in reports:
            print(f"{args.file} not found.", file=sys.stderr)
            return 1
    else:
        winner = max(reports, key=lambda f: reports[f][2]["ultra_frac"])

    sr, x, r = reports[winner]
    print("-" * 68)
    print(f"==> SUSPICIOUS FILE: {winner.name}  "
          f"({r['ultra_frac']*100:.2f}% ultrasonic energy, "
          f"carrier ≈ {r['carrier']:.0f} Hz)")

    print("\n" + "=" * 68)
    print(f"Step 3   : Demodulating carrier {r['carrier']:.0f} Hz -> audible band")
    print("=" * 68)
    code = demodulate(sr, x, r["carrier"])

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{winner.stem}_secret_code.wav"
    save_wav(out_path, sr, code)
    print(f"  Recovered code written to: {out_path}")

    print("\n" + "=" * 68)
    print("Step 4   : Playing the recovered four-number secret code")
    print("=" * 68)
    if args.no_play:
        print("  (--no-play set; open the WAV above to listen)")
    else:
        play(sr, code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
