import json
import os
import subprocess
import re

MANIFEST = "../splits/part_0_manifest.jsonl"
VIDEO_DIR = "videos_raw"
CLIP_DIR = "clips"

os.makedirs(CLIP_DIR, exist_ok=True)

def get_ytid(url):
    m = re.search(r"v=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None

count = 0
skip = 0

with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:

        sample = json.loads(line)

        url = sample["url"]
        start = float(sample["start_time"])
        end = float(sample["end_time"])
        label = sample["label"]
        text = sample["text"]
        signer = sample["signer_id"]

        ytid = get_ytid(url)

        if not ytid:
            continue

        video_path = os.path.join(VIDEO_DIR, f"{ytid}.mp4")

        if not os.path.exists(video_path):
            skip += 1
            continue

        out_name = f"l{label}_{text}_s{signer}_{ytid}_{start:.2f}_{end:.2f}.mp4"
        out_path = os.path.join(CLIP_DIR, out_name)

        if os.path.exists(out_path):
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c", "copy",
            out_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        count += 1

        if count % 50 == 0:
            print("processed", count)

print("done")
print("clips created:", count)
print("videos missing:", skip)