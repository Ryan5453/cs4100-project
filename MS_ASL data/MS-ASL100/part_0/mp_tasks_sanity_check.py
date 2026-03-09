import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = r"models\hand_landmarker.task"
VIDEO_PATH = r"clips\l0_hello_s0_FVjpLa8GqeM_0.00_1.74.mp4"  # 改成某个clip

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

def main():
  cap = cv2.VideoCapture(VIDEO_PATH)
  if not cap.isOpened():
    raise RuntimeError("Cannot open video")

  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
  if n <= 0:
    raise RuntimeError("No frames")

  # 均匀抽 10 帧
  picks = np.linspace(0, n - 1, 10).round().astype(int)

  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
  )

  found = False
  with HandLandmarker.create_from_options(options) as landmarker:
    for fi in picks:
      cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
      ok, frame_bgr = cap.read()
      if not ok:
        continue

      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

      ts = int(round(fi * 1000.0 / fps))
      result = landmarker.detect_for_video(mp_image, ts)

      num = 0 if not result.hand_landmarks else len(result.hand_landmarks)
      print(f"frame {fi}/{n-1}  ts={ts}ms  hands={num}")

      if num > 0:
        found = True
        # 顺便打印左右手
        if result.handedness:
          for i, h in enumerate(result.handedness):
            cat = h[0]
            print(f"  hand[{i}] => {cat.category_name} score={cat.score:.3f}")
        break

  cap.release()
  print("mediapipe:", mp.__version__)
  print("sanity:", "PASS" if found else "FAIL")

if __name__ == "__main__":
  main()