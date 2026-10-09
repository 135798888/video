"""Procedural soundtrack for the Giggle Academy reel: music bed + SFX + Chinese VO.

Every hit is placed on the composition's own animation times so picture and
sound stay locked. Deterministic (seeded noise). Usage:
    python3 make_soundtrack.py <audio_dir>
Writes <audio_dir>/soundtrack.wav (48 kHz stereo, 15.0 s).
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 15.0
N = int(SR * DUR)
rng = np.random.default_rng(20261009)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

music = np.zeros((N, 2))
sfx = np.zeros((N, 2))
vo = np.zeros(N)


def t_arr(d):
    return np.arange(int(SR * d)) / SR


def place(buf, sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    if buf.ndim == 1:
        buf[i : i + len(sig)] += sig
        return
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i : i + len(sig), 0] += sig * l * 1.414
    buf[i : i + len(sig), 1] += sig * r * 1.414


def onepole_lp(x, fc):
    fc = np.broadcast_to(np.asarray(fc, dtype=float), x.shape)
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    s = 0.0
    for i in range(len(x)):
        s = (1 - a[i]) * x[i] + a[i] * s
        y[i] = s
    return y


def hp(x):
    return np.diff(x, prepend=0.0)


def noise(d):
    return rng.uniform(-1, 1, int(SR * d))


def env_exp(d, k):
    return np.exp(-t_arr(d) * k)


def saw(f, d, detune=(0.0,)):
    t = t_arr(d)
    return sum(2 * ((t * f * (1 + dt)) % 1) - 1 for dt in detune) / len(detune)


def midi(n):
    return 440 * 2 ** ((n - 69) / 12)


# ---------------- music ----------------
BEAT = 0.45  # 133.3 BPM, grid starts on the drop at 1.6s
T0 = 1.6
END_MUSIC = 15.0


def kick():
    d = 0.4
    t = t_arr(d)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_exp(d, 9) + 0.3 * noise(d) * env_exp(d, 220)


def clap():
    d = 0.25
    n = hp(noise(d))
    e = env_exp(d, 18) * (1 + 0.6 * (t_arr(d) < 0.012))
    return onepole_lp(n * e, 5000) * 1.4


def hat(open_=False):
    d = 0.18 if open_ else 0.05
    return hp(hp(noise(d))) * env_exp(d, 20 if open_ else 90) * 0.5


K, C, H, HO = kick(), clap(), hat(), hat(True)

# chord progression (C  G  Am  F), one chord per bar of 4 beats
PROG = [[60, 64, 67, 72], [55, 62, 67, 71], [57, 60, 64, 69], [53, 60, 65, 69]]
ROOTS = [36, 31, 33, 29]

beats = []
b = T0
while b < END_MUSIC - 0.2:
    beats.append(b)
    b += BEAT

# keep drums out during the S5→S6 wipe fill and the final ring-out
for i, bt in enumerate(beats):
    in_fill = 12.3 <= bt < 12.6
    if bt >= 14.15:
        continue
    if not in_fill:
        place(music, K, bt, 0.9)
        if i % 2 == 1:
            place(music, C, bt, 0.45, 0.1)
    place(music, H, bt + BEAT / 2, 0.35, 0.3)
    if i % 4 == 3:
        place(music, HO, bt + BEAT / 2, 0.25, -0.3)

# fill: snare roll into the finale
for j in range(8):
    place(music, C, 12.25 + j * 0.045, 0.15 + j * 0.05, 0.0)

# sidechain envelope from the kick grid
side = np.ones(N)
for bt in beats:
    if bt >= 14.15:
        break
    i = int(bt * SR)
    d = int(0.3 * SR)
    seg = 1 - 0.65 * np.exp(-np.arange(d) / SR * 14)
    side[i : i + d] = np.minimum(side[i : i + d], seg[: N - i])

# bass + chord stabs
for i, bt in enumerate(beats):
    bar = (i // 4) % 4
    if bt >= 14.15:
        break
    root = midi(ROOTS[bar])
    d = BEAT * 0.95
    bass = saw(root, d, (0, 0.004)) + 0.6 * np.sin(2 * np.pi * root / 2 * t_arr(d))
    bass = onepole_lp(bass, 380) * np.minimum(1, t_arr(d) * 200) * env_exp(d, 3)
    place(music, bass, bt, 0.55)
    # offbeat stab
    chord = sum(saw(midi(n), 0.22, (-0.006, 0, 0.006)) for n in PROG[bar]) / 4
    chord = onepole_lp(chord, 2600) * env_exp(0.22, 14)
    place(music, chord, bt + BEAT / 2, 0.42, 0.25 if i % 2 else -0.25)

# playful 16th-note arpeggio through the data + UI scenes
ARP = [0, 2, 3, 1]
step = BEAT / 4
t = 6.0
k = 0
while t < 12.3:
    bar = (int((t - T0) / BEAT) // 4) % 4
    n = PROG[bar][ARP[k % 4]] + 12
    d = step * 0.9
    tt = t_arr(d)
    sq = np.sign(np.sin(2 * np.pi * midi(n) * tt)) * 0.5 + np.sin(2 * np.pi * midi(n) * 2 * tt) * 0.3
    place(music, onepole_lp(sq, 3500) * env_exp(d, 22), t, 0.16, 0.4 * np.sin(k * 0.7))
    t += step
    k += 1

# pad under the intro (C add9 swell) and final ring-out chord
def pad(notes, d, attack):
    tt = t_arr(d)
    s = sum(saw(midi(n), d, (-0.008, 0, 0.008)) for n in notes) / len(notes)
    s = onepole_lp(s, 1400)
    e = np.minimum(1, tt / attack)
    return s * e


place(music, pad([48, 55, 62, 64], 1.6, 1.4) * np.linspace(1, 0.6, int(1.6 * SR)), 0.0, 0.35)
ring = pad([48, 60, 64, 67, 72, 76], 0.86, 0.01) * env_exp(0.86, 2.2)
place(music, ring, 14.14, 0.5)
# bright bell arpeggio on the end lockup
for j, n in enumerate([72, 76, 79, 84]):
    d = 0.7
    tt = t_arr(d)
    bell = (np.sin(2 * np.pi * midi(n) * tt) + 0.4 * np.sin(2 * np.pi * midi(n) * 2.76 * tt)) * env_exp(d, 6)
    place(music, bell, 14.16 + j * 0.06, 0.22, -0.4 + j * 0.27)

music[:, 0] *= side
music[:, 1] *= side

# ---------------- SFX ----------------
def boing(f0, d=0.32):
    tt = t_arr(d)
    f = f0 * (1 + 1.4 * (1 - np.exp(-tt * 12))) * (1 + 0.04 * np.sin(2 * np.pi * 22 * tt))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(d, 9)


def thud(f0=90, d=0.18):
    tt = t_arr(d)
    f = f0 * (1 + 2 * np.exp(-tt * 40))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(d, 22) + 0.2 * onepole_lp(noise(d), 1800) * env_exp(d, 60)


def pop(f0=900, d=0.08):
    tt = t_arr(d)
    f = f0 * (1 + 1.5 * np.exp(-tt * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(d, 45)


def whoosh(d, up=True):
    tt = t_arr(d)
    shape = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 2
    fc = 600 + 5200 * (tt / d if up else 1 - tt / d)
    return onepole_lp(noise(d), fc) * shape * 1.6


def riser(d):
    tt = t_arr(d)
    x = onepole_lp(noise(d), 300 + 7000 * (tt / d) ** 2) * (tt / d) ** 1.5
    tone = np.sin(2 * np.pi * np.cumsum(200 + 900 * (tt / d) ** 2) / SR) * (tt / d) ** 2 * 0.3
    return (x + tone) * 1.3


def impact(d=1.0):
    tt = t_arr(d)
    f = 38 + 80 * np.exp(-tt * 18)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(d, 3.5)
    crack = onepole_lp(noise(d), 4000) * env_exp(d, 14)
    return sub * 1.1 + crack * 0.5


def tick(f=2400):
    d = 0.025
    return np.sin(2 * np.pi * f * t_arr(d)) * env_exp(d, 220)


def chime(n, d=0.5):
    tt = t_arr(d)
    f = midi(n)
    return (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 3.01 * tt) * np.exp(-tt * 10)) * env_exp(d, 7)


# S1: drop, bounces, burst, flood
place(sfx, whoosh(0.42, up=False), 0.0, 0.25)
place(sfx, boing(160), 0.42, 0.6)
place(sfx, thud(70), 0.42, 0.6)
place(sfx, boing(220), 1.01, 0.55)
place(sfx, thud(80), 1.01, 0.4)
place(sfx, riser(0.95), 0.65, 0.45)
place(sfx, impact(0.9), 1.2, 0.55)
for j in range(14):
    place(sfx, pop(700 + (j * 137) % 1600), 1.2 + j * 0.022, 0.18, ((j * 0.37) % 2) - 1)
place(sfx, whoosh(0.4), 1.22, 0.35)
place(sfx, impact(1.1), 1.6, 0.7)

# S2: letter slams, & spin, LEARN rise, strobe, exit
for j in range(3):
    place(sfx, thud(110 + j * 15), 1.62 + 0.07 * j + 0.35, 0.5, -0.4 + j * 0.4)
place(sfx, whoosh(0.5), 2.15, 0.3, 0.3)
for j in range(4):
    place(sfx, pop(1200 + j * 180), 2.25 + j * 0.05, 0.16, -0.3 + j * 0.2)
for j in range(4):
    place(sfx, tick(1800 + j * 300), 2.95 + j * 0.09, 0.25)
place(sfx, whoosh(0.45), 3.38, 0.5)
place(sfx, whoosh(0.5, up=False), 3.8, 0.4, -0.2)

# S3: face spin, eyes/mouth draw, letters bounce, giggle wobble
place(sfx, whoosh(0.5), 3.9, 0.3)
place(sfx, pop(600), 4.25, 0.3)
for j in range(6):
    place(sfx, boing(260 + j * 40, 0.18), 4.3 + 0.07 * j, 0.22, -0.5 + j * 0.2)
place(sfx, chime(84), 4.5, 0.3)
place(sfx, chime(88), 4.62, 0.25)
for j in range(8):
    place(sfx, pop(1500 + (j % 2) * 300, 0.05), 5.05 + j * 0.09, 0.1)

# S4: iris, count-ups, camera tracks, globe pops, tile flips, stamp
place(sfx, riser(0.45), 5.55, 0.35)
place(sfx, whoosh(0.5), 5.95, 0.45)
for tc, dur in [(6.25, 1.0), (7.6, 0.9), (8.8, 0.8), (13.42, 0.5)]:
    # count-up ticks decelerate with expo.out
    for j in range(16):
        u = j / 16
        at = tc + dur * (-np.log2(1 - u * 0.999)) / 10
        place(sfx, tick(2200 + j * 40), at, 0.14, 0.2)
for j in range(12):
    place(sfx, pop(500 + j * 60, 0.06), 6.2 + 0.035 * j, 0.1, -0.5 + j / 12)
place(sfx, whoosh(0.55), 7.3, 0.55, -0.3)
for j in range(14):
    place(sfx, pop(1100 + (j * 211) % 900), 7.75 + 0.035 * j, 0.12, ((j * 0.53) % 2) - 1)
place(sfx, whoosh(0.55), 8.55, 0.55, 0.3)
for j in range(8):
    place(sfx, thud(240 + j * 20, 0.08), 8.95 + 0.045 * j, 0.25, -0.5 + j / 8)
place(sfx, impact(0.8), 9.55, 0.65)

# S5: wipe, phone fly-in, tiles, callouts, tap, expand, stars
place(sfx, whoosh(0.45), 9.65, 0.5, 0.4)
place(sfx, whoosh(0.8, up=False), 9.8, 0.3)
for j in range(6):
    place(sfx, pop(900 + j * 120), 10.15 + 0.05 * j, 0.15)
for j in range(4):
    place(sfx, pop(500, 0.05), 10.5 + 0.16 * j, 0.12, -0.6 if j < 2 else 0.6)
place(sfx, tick(1400), 11.0, 0.4)
place(sfx, pop(320), 11.04, 0.35)
place(sfx, whoosh(0.45), 11.12, 0.3)
place(sfx, riser(0.5) * 0.5, 11.65, 0.3)
for j, n in enumerate([79, 84, 88]):
    place(sfx, chime(n), 11.98 + 0.09 * j, 0.3, -0.3 + j * 0.3)

# S6: bar wipe, words, billion, lockup
for j in range(5):
    place(sfx, thud(140 + j * 25, 0.12), 12.58 + 0.04 * j, 0.25, -0.6 + j * 0.3)
place(sfx, whoosh(0.4), 12.62, 0.35)
place(sfx, whoosh(0.36), 13.2, 0.35)
place(sfx, whoosh(0.3, up=False), 13.98, 0.4)
place(sfx, impact(0.9), 14.14, 0.75)
for j in range(10):
    place(sfx, pop(1300 + (j * 173) % 1000, 0.06), 14.15 + j * 0.03, 0.1, ((j * 0.41) % 2) - 1)

# ---------------- VO ----------------
def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, np.int16) / 32768.0
    else:
        a = np.frombuffer(raw, np.int32) / 2147483648.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    # resample to SR (linear)
    n = int(len(a) * SR / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


VO_TIMES = {"l1": 1.95, "l2": 4.35, "l3": 6.25, "l4": 9.95, "l5": 12.68}
for key, at in VO_TIMES.items():
    a = read_wav(f"{OUT}/vo/{key}.wav")
    a = a / (np.max(np.abs(a)) + 1e-9) * 0.9
    place(vo, a, at)

# duck music + SFX under VO
vo_env = onepole_lp(np.abs(vo), 8.0)
duck = 1 - 0.55 * np.clip(vo_env / (vo_env.max() * 0.35 + 1e-9), 0, 1)
duck = onepole_lp(duck, 6.0)

m_part = music * 0.55 * duck[:, None]
s_part = sfx * 0.6 * (0.6 + 0.4 * duck[:, None])
for k_, a_ in VO_TIMES.items():
    i0, i1 = int(a_ * SR), int((a_ + 1.2) * SR)
    r = lambda x: 20 * np.log10(np.sqrt(np.mean(x[i0:i1] ** 2)) + 1e-9)
    print(k_, "vo", round(r(vo * 0.95), 1), "music", round(r(m_part[:, 0]), 1), "sfx", round(r(s_part[:, 0]), 1))
mix = music * 0.55 * duck[:, None] + sfx * 0.6 * (0.6 + 0.4 * duck[:, None]) + vo[:, None] * 0.95

# fades + soft clip + normalize
fi = int(0.02 * SR)
mix[:fi] *= np.linspace(0, 1, fi)[:, None]
fo = int(0.35 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
mix = np.tanh(mix * 1.2) / np.tanh(1.2)
mix = mix / np.max(np.abs(mix)) * 0.7

pcm = (mix * 32767).astype("<i2")
with wave.open(f"{OUT}/soundtrack.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", f"{OUT}/soundtrack.wav", mix.shape[0] / SR, "s")
