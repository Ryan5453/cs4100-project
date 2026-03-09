import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Config
SEED = 42
INDEX_CSV = "features_index.csv"

BASE_DIR = os.getcwd()

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

# Quality filter
MIN_MAX_DETECT_RATE = 0.05 

# Model
HIDDEN = 128
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

# Dataset
class NpyFeatureDataset(Dataset):
  def __init__(self, df: pd.DataFrame):
    self.df = df.reset_index(drop=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    path = row["feature_file"]

    # resolve relative path if needed
    if not os.path.isabs(path):
      path = os.path.join(BASE_DIR, path)

    x = np.load(path).astype(np.float32)  
    y = int(row["label"])

    # torch tensors
    x = torch.from_numpy(x) 
    y = torch.tensor(y, dtype=torch.long)
    return x, y

# Model (GRU classifier)
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
    # x: (B, T, F)
    out, h = self.gru(x)     
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
  # Load index
  df = pd.read_csv(INDEX_CSV)
  # ensure expected columns
  needed = {"label", "feature_file", "left_detect_rate", "right_detect_rate"}
  missing = needed - set(df.columns)
  if missing:
    raise RuntimeError(f"features_index.csv missing columns: {missing}")

  # Convert detect rates to float
  df["left_detect_rate"] = df["left_detect_rate"].astype(float)
  df["right_detect_rate"] = df["right_detect_rate"].astype(float)
  df["max_detect_rate"] = df[["left_detect_rate", "right_detect_rate"]].max(axis=1)

  # Filter low-quality
  before = len(df)
  df = df[df["max_detect_rate"] >= MIN_MAX_DETECT_RATE].copy()
  after = len(df)
  print(f"filter detect_rate >= {MIN_MAX_DETECT_RATE}: {before} -> {after}")

  # Determine number of classes from labels present
  labels = sorted(df["label"].unique().tolist())
  num_classes = max(labels) + 1  # MS-ASL100 should be 100
  print("num_classes inferred:", num_classes)

  # Stratified split
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

  # Create model
  model = GRUClassifier(
    input_dim=126,
    hidden=HIDDEN,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  criterion = nn.CrossEntropyLoss()

  best_val = 0.0
  best_path = "best_gru.pt"

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
          "hidden": HIDDEN, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
          "min_detect_rate": MIN_MAX_DETECT_RATE
        }
      }, best_path)

  print("best_val_acc:", best_val)
  print("saved:", best_path)

  # Confusion matrix on val (optional)
  ckpt = torch.load(best_path, map_location=device)
  model.load_state_dict(ckpt["model_state_dict"])
  val_acc, ys, ps = evaluate(model, val_loader)
  cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
  print("final val_acc:", val_acc)
  print("confusion_matrix shape:", cm.shape)

if __name__ == "__main__":
  main()