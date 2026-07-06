#!/usr/bin/env python3
"""Regenerate the Exercise 3 report figures at print resolution.

Reuses the exact analysis/embedding logic of the two walkthrough notebooks
(app/Ex3_walkthrough.ipynb, app/Ex3_2_walkthrough.ipynb) so every figure and
every number printed here matches the notebooks' results.

Run from the repo root:

    uv run --project app report/make_figures.py

Writes PNGs into report/figures/ and prints the statistics quoted in
report.md.
"""
import math
import random
import warnings
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "Ex3_files"
FIGS = ROOT / "report" / "figures"
APP_OUT = ROOT / "app" / "out"
FIGS.mkdir(exist_ok=True)
APP_OUT.mkdir(exist_ok=True)

# print-friendly defaults: tufte text column is ~4.2in, fullwidth ~6.4in
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COL = (4.2, 2.5)
WIDE = (6.4, 2.6)


def save(fig, name):
    fig.tight_layout(pad=0.4)
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Task 3.1 — analysis (mirrors app/Ex3_walkthrough.ipynb)
# --------------------------------------------------------------------------

def load_wav(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    return sr, x


FILES = sorted(DATA.glob("Ex3_sound[1-4].wav"))
signals = {f.name: load_wav(f) for f in FILES}

ULTRA = 18000.0
rows = []
print("\n== 3.1 detection table ==")
print(f"{'file':<18}{'sr':>8}{'dur':>8}{'%>=18k':>9}{'peak':>9}")
for name, (sr, x) in signals.items():
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    m = (freq >= ULTRA) & (freq < sr / 2 - 1.0)
    frac = 100 * mag[m].sum() / (mag.sum() + 1e-12)
    peak = freq[m][np.argmax(mag[m])]
    rows.append((name, frac, peak))
    print(f"{name:<18}{sr:>8}{len(x)/sr:>7.2f}s{frac:>8.3f}%{peak:>8.0f}Hz")

suspect_name, _, carrier = max(rows, key=lambda r: r[1])
sr4, x4 = signals[suspect_name]

# refine carrier to the exact peak bin (as in the notebook)
mag4 = np.abs(np.fft.rfft(x4 * np.hanning(len(x4))))
freq4 = np.fft.rfftfreq(len(x4), 1 / sr4)
band = (freq4 > carrier - 2000) & (freq4 < carrier + 2000)
carrier = float(freq4[band][np.argmax(mag4[band])])
print(f"suspect: {suspect_name}   refined carrier: {carrier:.0f} Hz")

# --- fig-energy: ultrasonic energy per file (column width) ----------------
short = [r[0].replace("Ex3_", "").replace(".wav", "") for r in rows]
pcts = [r[1] for r in rows]
colors = ["crimson" if p == max(pcts) else "steelblue" for p in pcts]
fig, ax = plt.subplots(figsize=COL)
ax.bar(short, pcts, color=colors, width=0.62)
for i, p in enumerate(pcts):
    ax.text(i, p + 0.18, f"{p:.2f}%", ha="center", fontsize=7)
ax.set_ylabel("energy at/above 18 kHz (%)")
ax.set_ylim(0, max(pcts) * 1.18)
save(fig, "fig-energy.png")

# --- fig-amband: carrier + sidebands zoom (column width) ------------------
sel = (freq4 >= 15000) & (freq4 <= sr4 / 2)
fig, ax = plt.subplots(figsize=COL)
ax.plot(freq4[sel] / 1000, mag4[sel], lw=0.55, color="black")
ax.axvline(carrier / 1000, color="crimson", ls="--", lw=0.9,
           label=f"carrier {carrier / 1000:.1f} kHz")
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("|X(f)|")
ax.legend(frameon=False)
save(fig, "fig-amband.png")

# --- demodulation (identical to the notebook) ------------------------------

def demodulate(sr, x, fc, half_bw=2500.0, code_bw=3500.0):
    lo, hi = max(fc - half_bw, 100.0), min(fc + half_bw, sr / 2 - 100.0)
    bp = butter(6, [lo, hi], btype="band", fs=sr, output="sos")
    bandsig = sosfiltfilt(bp, x)
    env = np.abs(hilbert(bandsig))
    env -= env.mean()
    lp = butter(6, code_bw, btype="low", fs=sr, output="sos")
    code = sosfiltfilt(lp, env)
    return code / (np.max(np.abs(code)) + 1e-12)


code_sig = demodulate(sr4, x4, carrier)

# --- fig-spectro: stego spectrogram + recovered code (fullwidth pair) -----
fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
f, t, S = spectrogram(x4, fs=sr4, nperseg=2048, noverlap=1536)
pm = a1.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12),
                   shading="gouraud", cmap="magma", rasterized=True)
a1.axhline(18, color="cyan", ls=":", lw=0.9)
a1.set_title(f"{suspect_name}: AM burst above 18 kHz")
a1.set_xlabel("time (s)")
a1.set_ylabel("kHz")
f, t, S = spectrogram(code_sig, fs=sr4, nperseg=2048, noverlap=1536)
a2.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12),
              shading="gouraud", cmap="magma", rasterized=True)
a2.set_ylim(0, 4)
a2.set_title("demodulated code: speech harmonics")
a2.set_xlabel("time (s)")
a2.set_ylabel("kHz")
save(fig, "fig-spectro.png")

# --------------------------------------------------------------------------
# Task 3.2 — key-based LSB embedding (mirrors app/Ex3_2_walkthrough.ipynb)
# --------------------------------------------------------------------------

INPUT_WAV = DATA / "Ex3_sound5.wav"
STEGO_WAV = APP_OUT / "Ex3_sound5_embedded.wav"
MESSAGE = "It always seems impossible until it's done"
KEY = 20240703
N_LSB = 2
HEADER_BITS = 32


def _read_samples(path):
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        if params.sampwidth != 2:
            raise ValueError("16-bit PCM only")
        raw = w.readframes(params.nframes)
    return np.frombuffer(raw, dtype=np.int16).copy(), params


def _key_order(n, key):
    rng = random.Random(key)
    order = list(range(n))
    rng.shuffle(order)
    return order


def embed(in_path, out_path, message, key, n_lsb):
    samples, params = _read_samples(in_path)
    msg = message.encode("utf-8")
    bits = format(len(msg), f"0{HEADER_BITS}b") + "".join(
        format(b, "08b") for b in msg)
    n_slots = math.ceil(len(bits) / n_lsb)
    pos = np.array(_key_order(len(samples), key)[:n_slots], dtype=np.int64)
    groups = [bits[i:i + n_lsb].ljust(n_lsb, "0")
              for i in range(0, len(bits), n_lsb)]
    vals = np.array([int(g, 2) for g in groups], dtype=np.int32)
    work = samples.astype(np.int32)
    work[pos] = (work[pos] & ~((1 << n_lsb) - 1)) | vals
    with wave.open(str(out_path), "wb") as w:
        w.setparams(params)
        w.writeframes(work.astype(np.int16).tobytes())
    changed = int(np.count_nonzero(work != samples.astype(np.int32)))
    delta = int(np.max(np.abs(work - samples.astype(np.int32))))
    return samples, work.astype(np.int16), n_slots, changed, delta


def extract(path, key, n_lsb):
    samples, _ = _read_samples(path)
    mask = (1 << n_lsb) - 1
    order = _key_order(len(samples), key)

    def read_bits(k):
        return "".join(format(int(v) & mask, f"0{n_lsb}b")
                       for v in samples[order[:k]])

    hdr = read_bits(math.ceil(HEADER_BITS / n_lsb))[:HEADER_BITS]
    msg_len = int(hdr, 2)
    total = HEADER_BITS + msg_len * 8
    bits = read_bits(math.ceil(total / n_lsb))
    mb = bits[HEADER_BITS:HEADER_BITS + msg_len * 8]
    return bytes(int(mb[i:i + 8], 2) for i in range(0, len(mb), 8)).decode()


print("\n== 3.2 embedding ==")
cover, stego, slots, changed, delta = embed(
    INPUT_WAV, STEGO_WAV, MESSAGE, KEY, N_LSB)
print(f"samples used   : {slots} of {len(cover)}"
      f" ({100 * slots / len(cover):.3f}% of capacity)")
print(f"samples changed: {changed}  max |delta|: {delta}")
recovered = extract(STEGO_WAV, KEY, N_LSB)
print(f"extracted      : {recovered!r}")
print(f"round-trip OK  : {recovered == MESSAGE}")
try:
    extract(STEGO_WAV, KEY + 1, N_LSB)
    print("wrong key      : decoded (unexpected)")
except UnicodeDecodeError:
    print("wrong key      : UnicodeDecodeError — message not recoverable")

# --- fig-stego: waveform w/ modified samples + spectrum overlay -----------
diff_idx = np.nonzero(cover != stego)[0]
sr5, _ = wavfile.read(INPUT_WAV)
fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
a1.plot(cover, lw=0.3, color="steelblue", label="cover audio")
a1.scatter(diff_idx, stego[diff_idx], color="crimson", s=5, zorder=3,
           label=f"{len(diff_idx)} modified samples")
a1.set_title("stego waveform: key-scattered LSB changes")
a1.set_xlabel("sample index")
a1.set_ylabel("amplitude (int16)")
a1.legend(loc="lower right", frameon=True, framealpha=0.95,
          edgecolor="0.8", markerscale=2.2)
for sig, lbl, c, ls in ((cover, "cover", "steelblue", "-"),
                        (stego, "stego", "crimson", "--")):
    xm = sig.astype(np.float64) / 32768.0
    m = np.abs(np.fft.rfft(xm * np.hanning(len(xm))))
    fr = np.fft.rfftfreq(len(xm), 1 / sr5)
    a2.semilogy(fr / 1000, m + 1e-4, lw=0.5, color=c, ls=ls, label=lbl,
                alpha=0.8)
a2.set_title("cover vs. stego spectrum (overlaid)")
a2.set_xlabel("frequency (kHz)")
a2.set_ylabel("|X(f)|")
a2.legend(frameon=False)
save(fig, "fig-stego.png")

print("\nall figures written to report/figures/")
