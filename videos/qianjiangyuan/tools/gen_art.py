"""Procedural ink-wash art for 《一滴水的远方》. Deterministic (seeded).
Usage: python3 gen_art.py <out_dir>
"""
import sys, os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1110)


def save(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name), optimize=True)


def smooth_noise(h, w, scale, octaves=4, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        sh, sw = max(2, int(h / scale * 2**o)), max(2, int(w / scale * 2**o))
        g = r.random((sh, sw))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(im) / 255.0 * amp
        tot += amp
        amp *= 0.5
    return out / tot


def fbm1d(n, scale, octaves=5, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = max(2, int(n / scale * 2**o))
        pts = r.random(k + 1)
        xs = np.linspace(0, k, n)
        i = np.floor(xs).astype(int).clip(0, k - 1)
        t = xs - i
        t = t * t * (3 - 2 * t)
        x += (pts[i] * (1 - t) + pts[i + 1] * t) * amp
        tot += amp
        amp *= 0.5
    return x / tot


# ---------- paper ----------
W, H = 1920, 1080
base = np.array([243, 237, 225], float)
tone = smooth_noise(H, W, 500, 4, seed=1) - 0.5
fine = rng.normal(0, 1, (H, W))
fine = np.asarray(Image.fromarray(((fine * 18) + 128).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), float) - 128
paper = base[None, None, :] + tone[..., None] * 16 + fine[..., None] * 0.55
# fibers
fib = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(fib)
for _ in range(1400):
    x, y = rng.uniform(0, W), rng.uniform(0, H)
    a = rng.uniform(0, np.pi)
    L = rng.uniform(8, 46)
    d.line([(x, y), (x + np.cos(a) * L, y + np.sin(a) * L)], fill=int(rng.uniform(20, 60)), width=1)
fib = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.7)), float) / 255
paper -= fib[..., None] * 22
save(paper, "paper.png")

# paper overlay (for dark scenes / global texture): neutral grey modulation, RGBA
ov = np.zeros((H, W, 4))
ov[..., :3] = 128 + (tone[..., None] * 60 + fine[..., None] * 1.2 - fib[..., None] * 60)
ov[..., 3] = 255
save(ov, "paper_overlay.png")

# ---------- film grain (4 frames, half-res) ----------
for k in range(4):
    g = np.random.default_rng(50 + k).normal(128, 34, (540, 960))
    save(g, f"grain_{k}.png")


# ---------- mountains ----------
def mountain_layer(name, w, h, base_y, amp, scale, color, edge, seed, trees=0.0, blur=2.0, fade=0.75):
    ridge = base_y - fbm1d(w, scale, 6, seed) * amp
    # sharpen peaks a little
    ridge = ridge - (fbm1d(w, scale / 4, 3, seed + 7) - 0.5) * amp * 0.18
    yy = np.arange(h)[:, None]
    depth = (yy - ridge[None, :])
    inside = depth > 0
    # wash: dark near ridge, fading downward
    a = np.where(inside, np.exp(-depth / (h * fade * 0.32)) * 0.88 + 0.12 * np.exp(-depth / (h * 1.2)), 0.0)
    tex = smooth_noise(h, w, 160, 4, seed + 3)
    a = a * (0.72 + 0.5 * tex)
    # ink edge
    e = np.where(inside, np.exp(-depth / 5.0), 0.0) * edge
    a = np.clip(a + e, 0, 1)
    img = np.zeros((h, w, 4))
    col = np.array(color, float)
    img[..., :3] = col[None, None, :] * (1 - 0.25 * e[..., None])
    img[..., 3] = a * 255
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGBA")
    if trees > 0:
        dr = ImageDraw.Draw(im)
        r = np.random.default_rng(seed + 11)
        n = int(w * trees)
        for _ in range(n):
            x = r.uniform(0, w)
            yr = ridge[int(x) % w]
            y = yr + r.uniform(2, 60) ** 1.2
            s = r.uniform(2, 6)
            c = tuple(int(v * 0.42) for v in color) + (int(r.uniform(120, 210)),)
            dr.ellipse([x - s, y - s * 1.4, x + s, y + s * 0.6], fill=c)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    im.save(os.path.join(OUT, name), optimize=True)


mountain_layer("mtn_far.png", 2600, 1080, 600, 330, 700, (150, 170, 168), 0.25, 21, 0.0, 6, 0.55)
mountain_layer("mtn_mid.png", 2600, 1080, 760, 380, 520, (104, 122, 112), 0.5, 22, 0.25, 3, 0.6)
mountain_layer("mtn_near.png", 2600, 1080, 930, 330, 380, (52, 60, 54), 0.9, 23, 0.6, 1.6, 0.7)

# mist strip
mw, mh = 2600, 700
m = smooth_noise(mh, mw, 260, 4, 31)
yy = np.linspace(-1, 1, mh)[:, None]
band = np.exp(-(yy**2) / 0.18)
ma = np.clip((m - 0.38) * 2.2, 0, 1) * band
mist = np.zeros((mh, mw, 4))
mist[..., :3] = 250
mist[..., 3] = ma * 235
Image.fromarray(mist.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(8)).save(os.path.join(OUT, "mist.png"))

# ---------- ink splash for the climax ----------
S = 1400
sp = Image.new("L", (S, S), 0)
d = ImageDraw.Draw(sp)
c = S / 2
r = np.random.default_rng(77)
for i in range(70):
    ang = r.uniform(0, 2 * np.pi)
    dist = r.uniform(250, 640) ** 1.0
    rad = r.uniform(3, 22) * (1.2 - dist / 700)
    x, y = c + np.cos(ang) * dist, c + np.sin(ang) * dist
    d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=int(r.uniform(150, 255)))
    # streak toward center
    if r.random() < 0.5:
        d2 = dist - r.uniform(20, 70)
        d.line([(x, y), (c + np.cos(ang) * d2, c + np.sin(ang) * d2)], fill=int(r.uniform(90, 180)), width=max(1, int(rad / 2)))
sp = sp.filter(ImageFilter.GaussianBlur(1.6))
arr = np.zeros((S, S, 4))
arr[..., :3] = [38, 37, 35]
arr[..., 3] = np.asarray(sp, float)
Image.fromarray(arr.astype(np.uint8), "RGBA").save(os.path.join(OUT, "ink_splash.png"))

# soft ink ring (wash) used for ripples
R = 1200
yy, xx = np.mgrid[0:R, 0:R]
rr = np.hypot(xx - R / 2, yy - R / 2) / (R / 2)
nz = smooth_noise(R, R, 90, 3, 88)
ring = np.exp(-((rr - 0.86 - (nz - 0.5) * 0.06) ** 2) / 0.0016) * 0.9 + np.exp(-((rr - 0.8) ** 2) / 0.02) * 0.18 * (rr < 0.86)
ring = ring * (0.75 + 0.5 * nz)
arr = np.zeros((R, R, 4))
arr[..., :3] = [42, 46, 46]
arr[..., 3] = np.clip(ring, 0, 1) * 255
Image.fromarray(arr.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(1.2)).save(os.path.join(OUT, "ink_ring.png"))

# watercolor wash blob (generic soft stain)
B = 900
yy, xx = np.mgrid[0:B, 0:B]
rr = np.hypot(xx - B / 2, yy - B / 2) / (B / 2)
nz = smooth_noise(B, B, 120, 4, 99)
a = np.clip(1 - rr - (nz - 0.5) * 0.5, 0, 1) ** 1.4
edge = np.exp(-((1 - rr - (nz - 0.5) * 0.5) ** 2) / 0.002) * 0.35
arr = np.zeros((B, B, 4))
arr[..., :3] = 255
arr[..., 3] = np.clip(a * 0.7 + edge, 0, 1) * 255
Image.fromarray(arr.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(2)).save(os.path.join(OUT, "wash.png"))
print("ok")
