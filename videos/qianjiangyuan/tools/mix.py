"""Mix 《一滴水的远方》 soundtrack: the supplied music from 0s (untrimmed) + a very light
ambience bed + 8 short, clean SFX. Usage: python3 mix.py <project_dir> [--preview a b]
"""
import sys, subprocess, wave
import numpy as np

D = sys.argv[1]
SR = 48000
DUR = 160.1
N = int(round(DUR * SR))
rng = np.random.default_rng(2026)

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{D}/audio/music.mp3", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
music = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
mus = np.zeros((N, 2))
mus[: min(N, len(music))] = music[:N]

HIT = 511 / 30
BEAT = 0.49887
B = lambda k: 17.04 + k * BEAT


def t_(d):
    return np.arange(int(d * SR)) / SR


def lp(x, fc):
    # one-pole low-pass, vectorised via lfilter-like recursion in chunks
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    s = 0.0
    for i in range(len(x)):
        s = (1 - a) * x[i] + a * s
        y[i] = s
    return y


def lp_fast(x, fc):
    # FFT brick-ish low-pass with soft roll-off (for long noise beds)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X, len(x))


def hp_fast(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (fc / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, len(x))


def reverb(sig, decay=1.6, wet=0.35):
    L = int(decay * SR)
    ir = rng.normal(0, 1, L) * np.exp(-t_(decay) * 6.9 / decay)
    ir = lp_fast(ir, 5000)
    ir /= np.sqrt(np.sum(ir**2))
    n = len(sig) + L
    w = np.fft.irfft(np.fft.rfft(sig, n) * np.fft.rfft(ir, n), n)
    out = np.zeros(n)
    out[: len(sig)] += sig * (1 - wet)
    out += w * wet
    return out


sfx = np.zeros((N, 2))
amb = np.zeros((N, 2))


def place(buf, sig, at, gain_db, pan=0.0):
    i = int(round(at * SR))
    sig = sig[: max(0, N - i)] * 10 ** (gain_db / 20)
    l, r = np.cos((pan + 1) * np.pi / 4) * 1.414, np.sin((pan + 1) * np.pi / 4) * 1.414
    buf[i : i + len(sig), 0] += sig * l
    buf[i : i + len(sig), 1] += sig * r


def plink(f0=1400, d=0.35):
    t = t_(d)
    f = f0 * (1 + 0.9 * (1 - np.exp(-t * 60)))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18) * np.minimum(1, t * 3000)
    return reverb(s / np.max(np.abs(s)), 1.8, 0.4)


def impact():
    d = 1.6
    t = t_(d)
    f = 42 + 60 * np.exp(-t * 22)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    click = lp_fast(rng.normal(0, 1, len(t)), 1800) * np.exp(-t * 60) * 0.35
    s = (body + click) * np.minimum(1, t * 2000)
    return reverb(s / np.max(np.abs(s)), 2.2, 0.22)


def whoosh(d=0.9):
    t = t_(d)
    n = rng.normal(0, 1, len(t))
    env = np.sin(np.pi * t / d) ** 2
    s = lp_fast(hp_fast(n, 400), 3000) * env
    return s / np.max(np.abs(s))


def chime(freqs=(880, 1318.5, 1760), d=2.6):
    t = t_(d)
    s = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * (2.2 + i)) / (i + 1) for i, f in enumerate(freqs))
    s *= np.minimum(1, t * 800)
    return reverb(s / np.max(np.abs(s)), 2.4, 0.35)


def glass(d=1.4):
    t = t_(d)
    s = sum(np.sin(2 * np.pi * f * t + p) * np.exp(-t * k) for f, p, k in [(2100, 0, 6), (3170, 1, 8), (4630, 2, 10)])
    s *= np.minimum(1, t * 1500)
    return reverb(s / np.max(np.abs(s)), 1.8, 0.35)


# ---- 8 SFX (times match the picture) ----
CUES = [
    ("dew lands on the stream", 5.9, plink(1300), -21, 0.0),
    ("climax impact (downbeat)", HIT, impact(), -6.5, 0.0),
    ("ink ripple shimmer", HIT, plink(900, 0.5), -24, 0.0),
    ("drop into rock crack", 26.82, plink(1600), -22, 0.15),
    ("stream bends into road", B(90) - 0.2, whoosh(1.0), -27, -0.2),
    ("glass of water trembles", 90.6, glass(), -26, 0.0),
    ("lamp lights at home", B(181), chime(), -26, 0.1),
    ("title drop", B(268), plink(1100, 0.5), -20, 0.0),
]
for name, at, sig, g, pan in CUES:
    place(sfx, sig, at, g, pan)


# ---- ambience bed: very light, scene-aware ----
def env_seg(segments):
    e = np.zeros(N)
    for a, b, lvl in segments:
        i, j = int(a * SR), int(b * SR)
        fade = int(1.2 * SR)
        seg = np.full(j - i, lvl)
        k = min(fade, (j - i) // 2)
        seg[:k] *= np.linspace(0, 1, k)
        seg[-k:] *= np.linspace(1, 0, k)
        e[i:j] = np.maximum(e[i:j], seg)
    return e


white = rng.normal(0, 1, (N, 2))
# stream water: band-passed noise with slow amplitude flutter
water = hp_fast(lp_fast(white[:, 0], 2600), 500)
water2 = hp_fast(lp_fast(white[:, 1], 2600), 500)
flutter = 0.6 + 0.4 * lp_fast(np.abs(rng.normal(0, 1, N)), 6) / 0.8
water_st = np.stack([water * flutter, water2 * flutter], 1)
water_st /= np.std(water_st)
wind = lp_fast(white[:, 0] + white[:, 1], 380)
wind = np.stack([wind, wind], 1) / np.std(wind)
rain = hp_fast(white[:, 0] - white[:, 1], 1800)
rain = np.stack([rain, np.roll(rain, 997)], 1) / np.std(rain)
city = lp_fast(white[:, 0], 160)
city = np.stack([city, city], 1) / np.std(city)

e_water = env_seg([(4.0, 16.4, 0.7), (20.9, 41.5, 1.0), (41.5, 50.0, 0.7), (57.9, 62.4, 0.8), (93.6, 113, 1.0), (138.9, 151, 0.9)])
e_wind = env_seg([(0, 16.4, 0.6), (20.9, 60, 0.6), (112.4, 158, 0.6)])
e_rain = env_seg([(79.4, 94.0, 1.0)])
e_city = env_seg([(65.9, 79.9, 1.0), (0, 4.6, 0.7)])

lvl = lambda db: 10 ** (db / 20)
amb = water_st * e_water[:, None] * lvl(-44) + wind * e_wind[:, None] * lvl(-48) + rain * e_rain[:, None] * lvl(-40) + city * e_city[:, None] * lvl(-46)
# keep the 16.9s→hit breath clean: duck ambience to silence there
breath = np.ones(N)
i0, i1 = int(16.4 * SR), int(HIT * SR)
breath[i0:i1] = np.linspace(1, 0, i1 - i0)
breath[i1 : i1 + int(3.5 * SR)] = np.linspace(0, 1, int(3.5 * SR))
amb *= breath[:, None]

MUSIC_GAIN = lvl(-4.0)
mix = mus * MUSIC_GAIN + sfx + amb

# gentle peak control: soft-knee limiter on the sum (keeps dynamics, avoids clipping)
peak = np.max(np.abs(mix), axis=1)
ceil = lvl(-1.0)
g = np.ones(N)
over = peak > ceil * 0.8
g[over] = (ceil * 0.8 + (ceil * 0.2) * np.tanh((peak[over] - ceil * 0.8) / (ceil * 0.2))) / peak[over]
# smooth gain (attack fast, release ~80ms)
gs = np.minimum.accumulate(g[::-1])[::-1] if False else g
rel = np.exp(-1 / (0.08 * SR))
out_g = np.empty(N)
cur = 1.0
for i in range(N):
    cur = g[i] if g[i] < cur else g[i] + (cur - g[i]) * rel
    out_g[i] = cur
mix *= out_g[:, None]

pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
with wave.open(f"{D}/audio/mix.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("mix.wav", N / SR, "s; peak", float(np.max(np.abs(mix))))
for c in CUES:
    print(f"  SFX {c[1]:7.3f}s  {c[0]}")
