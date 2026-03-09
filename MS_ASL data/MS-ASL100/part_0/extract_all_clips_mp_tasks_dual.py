import os
import re
import csv
import time
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

MODEL_PATH = r"models\hand_landmarker.task"
CLIPS_DIR = "clips"
OUT_DIR = "features_mp"
INDEX_CSV = "features_index.csv"
FAILED_TXT = "failed_extract.txt"

T = 64
NUM_HANDS = 2

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

LABEL_RE = re.compile(r"^l(\d+)_")

def parse_label_from_filename(fn: str):
  m = LABEL_RE.match(fn)
  return int(m.group(1)) if m else None

def sample_indices(n_frames: int, t: int) -> np.ndarray:
  """Uniform sample if enough frames; else pad with last frame."""
  if n_frames <= 0:
    return np.zeros((t,), dtype=np.int32)
  if n_frames >= t:
    return np.linspace(0, n_frames - 1, t).round().astype(np.int32)
  base = np.arange(n_frames, dtype=np.int32)
  pad = np.full((t - n_frames,), n_frames - 1, dtype=np.int32)
  return np.concatenate([base, pad], axis=0)

def extract_one_clip(video_path: str, landmarker) -> tuple[np.ndarray, float, float]:
  """
  Returns:
    feat: (T, 126) float32
    left_rate: frames with Left detected / T
    right_rate: frames with Right detected / T
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError("Cannot open video")

  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
  picks = sample_indices(n_frames, T)

  feat = np.zeros((T, 126), dtype=np.float32)
  left_ok = 0
  right_ok = 0

  for ti, fi in enumerate(picks.tolist()):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ok, frame_bgr = cap.read()
    if not ok:
      continue

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # IMAGE 模式：不需要 timestamp
    result = landmarker.detect(mp_image)

    left = np.zeros((63,), dtype=np.float32)
    right = np.zeros((63,), dtype=np.float32)

    if result.hand_landmarks and result.handedness:
      for lm_list, handed_list in zip(result.hand_landmarks, result.handedness):
        handed = handed_list[0].category_name  # "Left" or "Right"

        pts = []
        for lm in lm_list:
          pts.extend([lm.x, lm.y, lm.z])
        arr = np.array(pts, dtype=np.float32)  # (63,)

        if handed == "Left":
          left = arr
          left_ok += 1
        elif handed == "Right":
          right = arr
          right_ok += 1

    feat[ti, :63] = left
    feat[ti, 63:] = right

  cap.release()
  return feat, left_ok / T, right_ok / T

def main():
  os.makedirs(OUT_DIR, exist_ok=True)

  clip_files = sorted([f for f in os.listdir(CLIPS_DIR) if f.lower().endswith(".mp4")])
  if not clip_files:
    raise RuntimeError(f"No mp4 found under {CLIPS_DIR}/")

  need_header = not os.path.exists(INDEX_CSV)

  # ✅ 改为 IMAGE 模式（核心修复点）
  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
  )

  processed = 0
  saved = 0
  skipped_existing = 0

  start_time = time.time()

  with HandLandmarker.create_from_options(options) as landmarker, \
       open(INDEX_CSV, "a", newline="", encoding="utf-8") as fcsv, \
       open(FAILED_TXT, "a", encoding="utf-8") as ff:

    w = csv.writer(fcsv)
    if need_header:
      w.writerow(["clip_file", "label", "feature_file", "left_detect_rate", "right_detect_rate"])

    for fn in tqdm(clip_files, desc="Extracting features"):
      stem = os.path.splitext(fn)[0]
      out_path = os.path.join(OUT_DIR, stem + ".npy")

      if os.path.exists(out_path):
        skipped_existing += 1
        continue

      label = parse_label_from_filename(fn)

      try:
        feat, lr, rr = extract_one_clip(os.path.join(CLIPS_DIR, fn), landmarker)
        np.save(out_path, feat)
        w.writerow([fn, label, out_path, f"{lr:.4f}", f"{rr:.4f}"])
        processed += 1
        saved += 1
      except Exception as e:
        ff.write(f"{fn}\t{repr(e)}\n")

  elapsed = time.time() - start_time
  print("\n==== DONE ====")
  print("clips total:", len(clip_files))
  print("new processed:", processed)
  print("saved:", saved)
  print("skipped existing:", skipped_existing)
  print(f"elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
  main()