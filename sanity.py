# eval_daytrain_from_cache.py
import os, json, time
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import config

# ================= CONFIG ================
CACHE_ROOT   = Path(config.CACHE_DIR) / "public_dayTrain"
IMAGES_DIR   = CACHE_ROOT / "images"
MASKS_DIR    = CACHE_ROOT / "masks"
MODEL_PATH   = "submissions/rge9ts_latest.pt"   # <-- set to your TorchScript file
DEVICE       = torch.device(config.DEVICE)
BATCH_SIZE   = 64                               # server-side inference BS (independent of student training)
LOG_EVERY    = 4000                             # print every ~N images during full-MSE
# =========================================

def banner(msg): print("\n" + "="*80 + f"\n{msg}\n" + "="*80)
def step(msg):   print(f"[STEP] {msg}")
def info(msg):   print(f"  - {msg}")

def _load_index(idx_path: Path) -> list[Dict[str, Any]]:
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index file: {idx_path}")
    with open(idx_path, "r") as f:
        return json.load(f)

def _list_shards(dir_path: Path, stem: str) -> list[Path]:
    return sorted(p for p in dir_path.glob(f"{stem}_shard_*.pt") if p.is_file())

# ---------- Iterable over cached image tensors ----------
class ImageShardIterable(IterableDataset):
    """
    Streams (C,H,W) tensors from image shards to avoid loading everything into memory.
    DataLoader will batch them into (B,C,H,W).
    """
    def __init__(self, images_dir: Path):
        super().__init__()
        self.images_dir = images_dir
        self.shards = _list_shards(images_dir, "images")
        if not self.shards:
            raise FileNotFoundError(f"No image shards found in {images_dir}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        for shard in self.shards:
            blob = torch.load(shard, map_location="cpu")
            X = blob["tensors"]  # (N,C,H,W)
            for i in range(X.size(0)):
                yield X[i]

# ---------- Iterable over cached (image, ROI-mask) pairs ----------
class ROIShardIterable(IterableDataset):
    """
    Streams (x, M) where:
      x: (C,H,W) from image shards
      M: (1,H,W) from mask shards
    We use the masks index as the driver (only annotated subset).
    """
    def __init__(self, images_dir: Path, masks_dir: Path):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        # indices
        self.img_index  = _load_index(images_dir / "images_index.json")
        self.mask_index = _load_index(masks_dir  / "masks_index.json")
        if not self.img_index or not self.mask_index:
            raise FileNotFoundError("images_index.json or masks_index.json is missing/empty.")

        # map path -> (shard_path, offset) for fast lookup
        self.img_loc: Dict[str, Tuple[Path, int]] = {}
        for rec in self.img_index:
            self.img_loc[rec["path"]] = (images_dir / rec["shard"], int(rec["offset"]))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # simple one-shard cache for images to reduce disk churn
        last_img_shard: Optional[Path] = None
        last_img_blob: Optional[torch.Tensor] = None

        for mrec in self.mask_index:
            m_shard = self.masks_dir / mrec["shard"]
            m_off   = int(mrec["offset"])
            # load M from its shard
            m_blob = torch.load(m_shard, map_location="cpu")["tensors"]   # (N,1,H,W)
            M = m_blob[m_off]  # (1,H,W)

            # find matching image tensor X
            path = mrec["path"]
            if path not in self.img_loc:
                # should not happen; skip gracefully
                continue
            img_shard_path, img_off = self.img_loc[path]

            if last_img_shard is None or img_shard_path != last_img_shard:
                last_img_blob = torch.load(img_shard_path, map_location="cpu")["tensors"]  # (N,C,H,W)
                last_img_shard = img_shard_path

            X = last_img_blob[img_off]  # (C,H,W)
            yield (X, M)

# ---------- Metrics ----------
MSE = nn.MSELoss(reduction="mean")

@torch.no_grad()
def eval_full_mse(model: torch.jit.ScriptModule, loader: DataLoader) -> float:
    step("Evaluating Full-image MSE on cached dayTrain …")
    total, n = 0.0, 0
    t0 = time.time()
    seen = 0
    for x in loader:                                # x: (B,C,H,W)
        x = x.to(DEVICE, non_blocking=True)
        y = model(x)
        total += MSE(y, x).item() * x.size(0)
        n += x.size(0)
        seen += x.size(0)
        if seen >= LOG_EVERY:
            info(f"Progress: {seen} images processed …")
            seen = 0
    dt = time.time() - t0
    info(f"Done Full-image MSE in {dt:.1f}s")
    return total / max(1, n)

@torch.no_grad()
def eval_roi_mse_per_image_mean(model: torch.jit.ScriptModule, loader: DataLoader) -> Tuple[float, int]:
    step("Evaluating ROI-MSE (per-image mean) on cached dayTrain …")
    scores, used = [], 0
    t0 = time.time()
    seen = 0
    for batch in loader:
        # DataLoader will collate into lists because tuples come from IterableDataset.
        # We convert to tensors and loop per-sample (annotated subset only).
        xs, Ms = batch
        # xs: list of T(C,H,W); Ms: list of T(1,H,W)
        for x, M in zip(xs, Ms):
            x = x.unsqueeze(0).to(DEVICE)          # (1,C,H,W)
            y = model(x)
            se = ((y - x) ** 2).mean(dim=1, keepdim=True).cpu()[0, 0]  # (H,W) channel-avg BEFORE masking
            den = float(M.sum().item())
            if den <= 0:
                continue
            roi = float((se * M[0]).sum().item() / den)
            scores.append(roi); used += 1
            seen += 1
            if seen % 1000 == 0:
                info(f"ROI progress: {seen} images …")
    dt = time.time() - t0
    info(f"Done ROI-MSE in {dt:.1f}s (used {used} annotated images)")
    if not scores:
        return float("nan"), 0
    return float(sum(scores) / len(scores)), used

def main():
    banner("Eval (dayTrain) from CACHE — Verbose Mode")

    # Sanity: cache presence
    step("Checking cache presence …")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing images cache dir: {IMAGES_DIR}")
    if not MASKS_DIR.exists():
        info("No masks cache dir found — ROI-MSE will be unavailable.")

    # Load model
    step("Loading TorchScript model …")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval()
    info(f"Device        : {DEVICE}")
    info(f"Model path    : {MODEL_PATH}")
    info(f"Cache root    : {CACHE_ROOT}")

    # Full dataset (all dayTrain images)
    step("Preparing cached full-image stream …")
    ds_full = ImageShardIterable(IMAGES_DIR)
    ld_full = DataLoader(ds_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    info("Cached full-image DataLoader ready.")

    # Full MSE
    full_mse = eval_full_mse(model, ld_full)

    # ROI dataset (annotated subset)
    if MASKS_DIR.exists() and (MASKS_DIR / "masks_index.json").exists():
        step("Preparing cached ROI stream …")
        ds_roi = ROIShardIterable(IMAGES_DIR, MASKS_DIR)
        # For ROI we want per-image evaluation; use batch_size that collates lists comfortably.
        ld_roi = DataLoader(ds_roi, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda b: (
            [x for x, m in b], [m for x, m in b]
        ))
        roi_mse, roi_n = eval_roi_mse_per_image_mean(model, ld_roi)
    else:
        roi_mse, roi_n = float("nan"), 0
        info("Skipping ROI-MSE (no cached masks found).")

    # Results
    banner("Results (dayTrain from cache)")
    print(f"Full-image MSE : {full_mse:.6f}  (cached images)")
    if roi_n > 0:
        print(f"ROI-MSE        : {roi_mse:.6f}  (per-image mean over {roi_n} annotated cached images)")
    else:
        print("ROI-MSE        : N/A (no annotated cache)")

if __name__ == "__main__":
    main()
