import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
SEED = 42
INDEX_CSV = "features_index.csv"
BASE_DIR = os.getcwd()

BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3

# TO-DO: need to change after get more data
MIN_MAX_DETECT_RATE = 0.0

# Model
INPUT_DIM = 252   
HIDDEN = 192
NUM_LAYERS = 1
DROPOUT = 0.0

# Reproducibility
def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# Hand feature utilities
def normalize_one_hand(hand_feat: np.ndarray) -> np.ndarray:
  """
  hand_feat: (T, 63)
  reshape -> (T, 21, 3)

  normalize per frame:
  1) subtract wrist (landmark 0)
  2) divide by hand scale

  return: (T, 63)
  """
  T = hand_feat.shape[0]
  hand = hand_feat.reshape(T, 21, 3).copy()

  # detect empty frames: all zeros
  is_empty = np.all(np.isclose(hand, 0.0), axis=(1, 2))

  for t in range(T):
    if is_empty[t]:
      continue

    pts = hand[t]  

    # wrist as origin
    wrist = pts[0].copy()
    pts = pts - wrist

    # scale: use max norm over all landmarks, avoid divide by zero
    norms = np.linalg.norm(pts, axis=1)
    scale = np.max(norms)

    if scale < 1e-6:
      scale = 1.0

    pts = pts / scale
    hand[t] = pts

  return hand.reshape(T, 63).astype(np.float32)


def normalize_dual_hand(x: np.ndarray) -> np.ndarray:
  """
  x: (T, 126)
  left:  0:63
  right: 63:126

  returns normalized position feature: (T,126)
  """
  left = x[:, :63]
  right = x[:, 63:]

  left_n = normalize_one_hand(left)
  right_n = normalize_one_hand(right)

  return np.concatenate([left_n, right_n], axis=1).astype(np.float32)


def compute_velocity(x: np.ndarray) -> np.ndarray:
  """
  x: (T, F)
  velocity[0] = 0
  velocity[t] = x[t] - x[t-1]
  """
  v = np.zeros_like(x, dtype=np.float32)
  v[1:] = x[1:] - x[:-1]
  return v

# Dataset
class NpyFeatureDataset(Dataset):
  def __init__(self, df: pd.DataFrame):
    self.df = df.reset_index(drop=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    path = row["feature_file"]

    if not os.path.isabs(path):
      path = os.path.join(BASE_DIR, path)

    x = np.load(path).astype(np.float32)  
    y = int(row["label"])

    x_norm = normalize_dual_hand(x)        

    x_vel = compute_velocity(x_norm)       

    x_feat = np.concatenate([x_norm, x_vel], axis=1).astype(np.float32)

    x_feat = torch.from_numpy(x_feat)
    y = torch.tensor(y, dtype=torch.long)

    return x_feat, y

# Model
class GRUClassifier(nn.Module):
  def __init__(self, input_dim: int, hidden: int, num_layers: int, num_classes: int, dropout: float):
    super().__init__()
    self.gru = nn.GRU(
      input_size=input_dim,
      hidden_size=hidden,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers > 1 else 0.0,
      bidirectional=False,
    )
    self.head = nn.Sequential(
      nn.LayerNorm(hidden),
      nn.Linear(hidden, num_classes)
    )

  def forward(self, x):
    # x: (B,T,F)
    out, _ = self.gru(x)
    last = out[:, -1, :]
    logits = self.head(last)
    return logits

# Train / Eval
@torch.no_grad()
def evaluate(model, loader):
  model.eval()
  ys = []
  ps = []

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    pred = torch.argmax(logits, dim=1)

    ys.extend(y.cpu().tolist())
    ps.extend(pred.cpu().tolist())

  acc = accuracy_score(ys, ps)
  return acc, ys, ps


def train_one_epoch(model, loader, optimizer, criterion):
  model.train()
  total_loss = 0.0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * x.size(0)

  return total_loss / len(loader.dataset)


def main():
  df = pd.read_csv(INDEX_CSV)

  needed = {"label", "feature_file", "left_detect_rate", "right_detect_rate"}
  missing = needed - set(df.columns)
  if missing:
    raise RuntimeError(f"features_index.csv missing columns: {missing}")

  df["left_detect_rate"] = df["left_detect_rate"].astype(float)
  df["right_detect_rate"] = df["right_detect_rate"].astype(float)
  df["max_detect_rate"] = df[["left_detect_rate", "right_detect_rate"]].max(axis=1)

  before = len(df)
  df = df[df["max_detect_rate"] >= MIN_MAX_DETECT_RATE].copy()
  after = len(df)
  print(f"filter detect_rate >= {MIN_MAX_DETECT_RATE}: {before} -> {after}")

  labels = sorted(df["label"].unique().tolist())
  num_classes = max(labels) + 1
  print("num_classes inferred:", num_classes)

  train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
  )

  train_ds = NpyFeatureDataset(train_df)
  val_ds = NpyFeatureDataset(val_df)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

  model = GRUClassifier(
    input_dim=INPUT_DIM,
    hidden=HIDDEN,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  criterion = nn.CrossEntropyLoss()

  best_val = 0.0
  best_path = "best_gru_norm_vel.pt"

  for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_acc, _, _ = evaluate(model, val_loader)

    print(f"epoch {epoch:02d} | loss {loss:.4f} | val_acc {val_acc:.4f}")

    if val_acc > best_val:
      best_val = val_acc
      torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "config": {
          "input_dim": INPUT_DIM,
          "hidden": HIDDEN,
          "num_layers": NUM_LAYERS,
          "dropout": DROPOUT,
          "min_detect_rate": MIN_MAX_DETECT_RATE
        }
      }, best_path)

  print("best_val_acc:", best_val)
  print("saved:", best_path)

  ckpt = torch.load(best_path, map_location=device)
  model.load_state_dict(ckpt["model_state_dict"])
  val_acc, ys, ps = evaluate(model, val_loader)
  cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
  print("final val_acc:", val_acc)
  print("confusion_matrix shape:", cm.shape)


if __name__ == "__main__":
  main()