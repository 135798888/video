"""Verify a render: picture/sound sync at the climax, black frames, audio presence.
Usage: python3 verify.py <video.mp4>
"""
import sys, subprocess, json
import numpy as np

V = sys.argv[1]


def probe():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels,duration:format=duration", "-of", "json", V], capture_output=True, text=True).stdout
    return json.loads(out)


info = probe()
print(json.dumps(info, ensure_ascii=False, indent=1))

# luma per frame 15.0–19.0s (frames 450–570)
W, H = 192, 108
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "15", "-i", V, "-t", "4", "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
luma = fr.reshape(len(fr), -1).mean(1)
f0 = 450
jump = int(np.argmax(np.diff(luma))) + 1
print("frames 500..520 luma:", [round(float(x), 1) for x in luma[50:71]])
lit_frame = f0 + jump
print("picture lights at frame", lit_frame, "=", round(lit_frame / 30, 4), "s")
still = luma[(504 - f0) : (511 - f0)]
print("16.8–17.0 luma range (should be ~flat):", round(float(still.max() - still.min()), 2))

# audio onset in the rendered file
SR = 48000
a = subprocess.run(["ffmpeg", "-v", "error", "-i", V, "-ss", "16.5", "-t", "1.0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(a, np.float32)
hop = int(SR * 0.002)
e = np.array([np.sqrt(np.mean(x[i : i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
db = 20 * np.log10(e + 1e-9)
k = int(np.argmax(np.diff(db) > 8))  # first strong rise
onset = 16.5 + (k + 1) * 0.002
print("audio onset at", round(onset, 4), "s = frame", round(onset * 30, 2))
print("sync error (frames):", round(lit_frame - onset * 30, 2))

# black / blank frame scan across the film at 2 fps
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", V, "-vf", "fps=2,scale=96:54,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
g = np.frombuffer(raw, np.uint8).reshape(-1, 54, 96)
m = g.reshape(len(g), -1).mean(1)
print("darkest samples (t, luma):", [(i / 2, round(float(m[i]), 1)) for i in np.argsort(m)[:6]])

# audio presence: first 2 s and last 3 s loudness
for t0, d in [(0, 2), (157, 3), (155, 2)]:
    a = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-i", V, "-t", str(d), "-ac", "1", "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(a, np.float32)
    print(f"audio rms {t0}-{t0 + d}s:", round(float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)), 1), "dBFS")
