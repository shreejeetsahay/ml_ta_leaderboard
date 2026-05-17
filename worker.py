# worker.py
import os
import threading
import queue
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models import SessionLocal, Result
from precompute_cache import (
    load_cached_public,   # must return {"full_loader": DataLoader, "roi_dataset": Dataset|None}
    load_cached_private,  # same shape as above
)

MIN_LD = getattr(config, "MIN_LATENT_DIM", 8)


# -----------------------------
# Thread-safe submission queue
# -----------------------------
# Thread-safe submission queue (cap at 60)
submission_queue = queue.Queue()

# def enqueue_submission(team_id: int, model_path: str) -> bool:
#     """
#     Try to enqueue without blocking. Returns True if queued, False if the queue is full.
#     """
#     try:
#         submission_queue.put_nowait((team_id, model_path))
#         print(f"[enqueue] queued team={team_id} {model_path} (size={submission_queue.qsize()})")
#         return True
#     except queue.Full:
#         print(f"[enqueue] queue full (limit={submission_queue.maxsize}). "
#               f"Rejecting team={team_id} {model_path}")
#         return False

# def enqueue_submission_block_safe(team_id: int, model_path: str) -> bool:
#     if submission_queue.full():
#         print(f"[enqueue] queue full with 60 submissions; rejecting team={team_id}")
#         return False
#     return enqueue_submission(team_id, model_path)


def worker():
    while True:
        team_id, model_path = submission_queue.get()
        try:
            print(f"[worker] Processing team {team_id} submission: {model_path}")
            evaluate_submission(team_id, model_path)
        except Exception as e:
            print(f"[worker] Fatal error while evaluating team {team_id}: {e}")
        finally:
            submission_queue.task_done()

thread = threading.Thread(target=worker, daemon=True)
thread.start()

# -----------------------------
# Metric helpers (fast, cache-ready)
# -----------------------------
MSE = nn.MSELoss(reduction="mean")

@torch.no_grad()
def reconstruct_via_enc_dec(
    model: torch.jit.ScriptModule,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Enforce the intended AE interface for grading:
      z = model.enc(x)
      y = model.dec(z)
    - requires model.enc and model.dec to be callable
    - requires enc(x) to return a 2D latent tensor (B, ld)
    """
    model.eval()

    enc = getattr(model, "enc", None)
    dec = getattr(model, "dec", None)
    if enc is None or not callable(enc) or dec is None or not callable(dec):
        raise RuntimeError("Model must define callable enc and dec for grading.")

    x = x.to(device, non_blocking=True)
    z = enc(x)

    if not isinstance(z, torch.Tensor):
        raise RuntimeError(f"enc(x) must return a Tensor latent, got {type(z)}")

    # Enforce true bottleneck vector: (B, latent_dim)
    if z.dim() != 2:
        raise RuntimeError(f"enc(x) must return a 2D tensor (B, latent_dim), got shape {tuple(z.shape)}")

    y = dec(z)
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"dec(z) must return a Tensor, got {type(y)}")

    return y

@torch.no_grad()
def eval_full_mse(model: torch.jit.ScriptModule, loader: DataLoader, device: torch.device):
    """Average pixel MSE over the entire dataset (already resized to 256x256 in cache)."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        # cached loader may yield x or (x, path)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        # reconstruction MUST go through enc/dec
        y = reconstruct_via_enc_dec(model, x, device)
        x = x.to(device, non_blocking=True)
        total += MSE(y, x).item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)

# @torch.no_grad()
# def eval_full_mse(model: torch.jit.ScriptModule, loader: DataLoader, device: torch.device):
#     """Average pixel MSE over the entire dataset (already resized to 256x256 in cache)."""
#     model.eval()
#     total, n = 0.0, 0
#     for batch in loader:
#         # cached loader may yield x or (x, path)
#         x = batch[0] if isinstance(batch, (tuple, list)) else batch
#         x = x.to(device, non_blocking=True)
#         y = model(x)
#         total += MSE(y, x).item() * x.size(0)
#         n += x.size(0)
#     return total / max(1, n)

@torch.no_grad()
def eval_roi_mse_per_image_mean(model: torch.jit.ScriptModule, roi_ds, device: torch.device):
    """
    roi_ds must yield either:
      (x, M)        or
      (x, M, path)
    where:
      x: (C,H,W) in [0,1]
      M: (1,H,W) binary mask for ROI (1 inside boxes), same H,W.
    Returns: (mean_roi_mse, num_images_used)
    """
    if roi_ds is None or len(roi_ds) == 0:
        return float("nan"), 0

    model.eval()
    scores, used = [], 0
    for i in range(len(roi_ds)):
        sample = roi_ds[i]
        if isinstance(sample, (tuple, list)):
            if len(sample) == 3:
                x, M, _ = sample
            else:
                x, M = sample
        else:
            # unexpected shape
            continue

        x = x.unsqueeze(0)  # (1,C,H,W)
        y = reconstruct_via_enc_dec(model, x, device)  # STRICT path

        x = x.to(device, non_blocking=True)
        se = ((y - x) ** 2).mean(dim=1, keepdim=True)  # (1,1,H,W)
        M_t = M.to(se.device)
        if M_t.dim() == 3:     # (1,H,W)
            M_t = M_t.unsqueeze(0)  # (1,1,H,W)
        elif M_t.dim() == 2:   # (H,W)
            M_t = M_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        den = float(M_t.sum().item())
        if den <= 0:
            continue
        roi = float((se * M_t).sum().item() / den)
        scores.append(roi)
        used += 1

    if not scores:
        return float("nan"), 0
    return float(sum(scores) / len(scores)), used

# @torch.no_grad()
# def eval_roi_mse_per_image_mean(model: torch.jit.ScriptModule, roi_ds, device: torch.device):
#     """
#     roi_ds must yield either:
#       (x, M)        or
#       (x, M, path)
#     where:
#       x: (C,H,W) in [0,1]
#       M: (1,H,W) binary mask for ROI (1 inside boxes), same H,W.
#     Returns: (mean_roi_mse, num_images_used)
#     """
#     if roi_ds is None or len(roi_ds) == 0:
#         return float("nan"), 0

#     model.eval()
#     scores, used = [], 0
#     for i in range(len(roi_ds)):
#         sample = roi_ds[i]
#         if isinstance(sample, (tuple, list)):
#             if len(sample) == 3:
#                 x, M, _ = sample
#             else:
#                 x, M = sample
#         else:
#             # unexpected shape
#             continue

#         x = x.unsqueeze(0).to(device)  # (1,C,H,W)
#         y = model(x)
#         se = ((y - x) ** 2).mean(dim=1, keepdim=True)  # (1,1,H,W)
#         M_t = M.to(se.device)
#         if M_t.dim() == 3:     # (1,H,W)
#             M_t = M_t.unsqueeze(0)  # (1,1,H,W)
#         elif M_t.dim() == 2:   # (H,W)
#             M_t = M_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

#         den = float(M_t.sum().item())
#         if den <= 0:
#             continue
#         roi = float((se * M_t).sum().item() / den)
#         scores.append(roi)
#         used += 1

#     if not scores:
#         return float("nan"), 0
#     return float(sum(scores) / len(scores)), used

# -----------------------------
# Latent dim extraction (best-effort)
# -----------------------------
def infer_latent_dim(scripted_model) -> int | None:
    """Try a few common places students might store the latent dimension."""
    try:
        for name in ("latent_dim", "LD", "ld"):
            if hasattr(scripted_model, name):
                return int(getattr(scripted_model, name))
    except Exception:
        pass

    try:
        enc = getattr(scripted_model, "enc", None)
        if enc is not None:
            for name in ("latent_dim", "LD", "ld"):
                if hasattr(enc, name):
                    return int(getattr(enc, name))
            fc = getattr(enc, "fc", None)
            if fc is not None and hasattr(fc, "out_features"):
                return int(fc.out_features)
    except Exception:
        pass

    return None

# @torch.no_grad()
# def infer_latent_dim_via_enc(
#     model: torch.jit.ScriptModule,
#     loader: DataLoader,
#     device: torch.device,
# ) -> int | None:
#     """Return ld by calling the encoder once and reading z.shape[1]."""
#     model.eval()
#     try:
#         batch = next(iter(loader))
#     except StopIteration:
#         return None
#     x = batch[0] if isinstance(batch, (tuple, list)) else batch
#     x = x.to(device, non_blocking=True)

#     # Expect AE.enc to exist per the starter.
#     enc = getattr(model, "enc", None)
#     if enc is None or not callable(enc):
#         return None

#     try:
#         z = enc(x)  # triggers Enc._init_fc on first call (expected)
#     except Exception:
#         return None

#     if isinstance(z, torch.Tensor):
#         # Typical case: z is (B, ld)
#         if z.dim() == 2:
#             return int(z.size(1))
#         # If someone made z spatial, flatten per sample
#         if z.dim() >= 3:
#             return int(z.view(z.size(0), -1).size(1))
#     return None

@torch.no_grad()
def infer_latent_dim_via_enc(
    model: torch.jit.ScriptModule,
    loader: DataLoader,
    device: torch.device,
) -> int | None:
    """Return ld by calling the encoder once and reading z.shape[1]."""
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.to(device, non_blocking=True)

    # Expect AE.enc to exist per the starter.
    enc = getattr(model, "enc", None)
    if enc is None or not callable(enc):
        return None

    try:
        z = enc(x)  # triggers Enc._init_fc on first call (expected)
    except Exception:
        return None

    if isinstance(z, torch.Tensor) and z.dim() == 2:
        # Typical case: z is (B, ld)
        return int(z.size(1))

    # Anything else (tuples, spatial tensors, etc.) is considered invalid
    return None


# -----------------------------
# Main evaluation entry
# -----------------------------
def evaluate_submission(team_id: int, model_path: str):
    """
    Loads TorchScript, evaluates on cached public/private splits, and updates the latest
    'pending' Result row for this team. Mirrors private ROI-MSE into legacy 'score'.
    """
    try:
        device = torch.device(config.DEVICE)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        try:
            pub_cache = load_cached_public(
                config.CACHE_DIR,
                batch_size=getattr(config, "BATCH_SIZE", 64),
                device=device,
            )
            prv_cache = load_cached_private(
                config.CACHE_DIR,
                batch_size=getattr(config, "BATCH_SIZE", 64),
                device=device,
            )
            # basic presence checks
            if not pub_cache or "full_loader" not in pub_cache:
                raise RuntimeError("Public cache missing 'full_loader'")
            if not prv_cache or "full_loader" not in prv_cache:
                raise RuntimeError("Private cache missing 'full_loader'")
            cache_status = True
        except Exception as e:
            cache_status = False

        with SessionLocal() as session:
            # Most recent pending attempt
            row = (
                session.query(Result)
                .filter_by(team_id=team_id)
                .order_by(Result.attempt.desc())
                .first()
            )
            if not row:
                print(f"[worker] No pending Result row for team_id={team_id}. Skipping.")
                return {"error": "no pending result row"}

            # Guard: file too large
            if size_mb > float(getattr(config, "MAX_MODEL_SIZE", 23.0)):  # <-- FIXED NAME
                row.status = "rejected (file too large)"
                row.model_size = size_mb
                session.commit()
                print(f"[worker] Rejected: model {size_mb:.2f} MB exceeds limit.")
                return {"error": "model too large"}

            # Load TorchScript
            try:
                model = torch.jit.load(model_path, map_location=device)
                model.eval()
            except Exception as e:
                row.status = "broken file"
                row.model_size = size_mb
                session.commit()
                print(f"[worker] TorchScript load failed: {e}")
                return {"error": str(e)}

            # Model props
            # row.latent_dim = (lambda v: int(v) if v is not None else None)(infer_latent_dim(model))
            row.img_size   = 256
            row.grayscale  = False
            row.model_size = size_mb
            row.artifact   = os.path.basename(model_path)

            # ----- Load cached datasets -----
            if cache_status == False:
                row.status = "internal cache error"
                session.commit()
                print(f"[worker] Cache load failed")
                return {"error": "cache error internal"}

            try:
                ld = infer_latent_dim_via_enc(model, pub_cache["full_loader"], device)
                if ld is not None:
                    row.latent_dim = int(ld)
                else:
                    raise RuntimeError("Could not infer latent_dim via encoder. Make sure your latent vector is of B X latent_dim shape")
            except RuntimeError as e:
                row.status = "Could not infer latent_dim via encoder. Make sure your latent vector is of B X latent_dim shape"
                session.commit()
                print(f"[worker] latent_dim inference failed: {e}")
                return {"error": str(e)}
            # Guard: latent_dim too small
            if row.latent_dim is None or row.latent_dim < MIN_LD:
                row.status = f"rejected (latent_dim < {MIN_LD})"
                session.commit()
                print(f"[worker] Rejected: latent_dim={row.latent_dim} < {MIN_LD}")
                return {"error": f"latent_dim < {MIN_LD}"}
            # ----- Evaluate PUBLIC (dayTrain) -----
            try:
                # after loading pub_cache/prv_cache and before setting row.latent_dim
                pub_full = eval_full_mse(model, pub_cache["full_loader"], device)
            except Exception as e:
                pub_full = float("nan")
                print("[worker] Public Full-MSE eval failed")
                row.status = "public full_mse eval failed with error:" + str(e)
                session.commit()
                return {"error": str(e)}
            
            if pub_full == 0.0:
                row.status = "rejected (zero public full_mse). This is a trivial solution."
                session.commit()
                print(f"[worker] Rejected: public full_mse is exactly zero.")
                return {"error": "public full_mse is exactly zero"}

            try:
                pub_roi, pub_roi_n = eval_roi_mse_per_image_mean(model, pub_cache.get("roi_dataset"), device)
            except Exception as e:
                pub_roi, pub_roi_n = float("nan"), 0
                print("[worker] Public ROI-MSE eval failed")
                row.status = "public roi_mse eval failed with error:" + str(e)
                session.commit()
                return {"error": str(e)}

            row.public_full_mse = pub_full
            row.public_roi_mse  = pub_roi
            row.public_roi_n    = pub_roi_n

            # ----- Evaluate PRIVATE (daySequence1+2) -----
            # try:
            #     prv_full = eval_full_mse(model, prv_cache["full_loader"], device)
            # except Exception as e:
                # prv_full = float("nan")
                # print("[worker] Private Full-MSE eval failed")
                # row.status = "private full_mse eval failed"
                # session.commit()
                # return {"error": str(e)}

            # try:
            #     prv_roi, prv_roi_n = eval_roi_mse_per_image_mean(model, prv_cache.get("roi_dataset"), device)
            # except Exception as e:
            #     prv_roi, prv_roi_n = float("nan"), 0
            #     print("[worker] Private ROI-MSE eval failed")
            #     row.status = "private roi_mse eval failed"
            #     session.commit()
            #     return {"error": str(e)}
            row.private_full_mse = 0.0
            row.private_roi_mse  = 0.0
            row.private_roi_n    = 0.0

            # Legacy compatibility
            row.score = 0.0

            # Mark success and commit
            row.status = "successful"
            if row.submitted_at and row.submitted_at.tzinfo is None:
                row.submitted_at = row.submitted_at.replace(tzinfo=ZoneInfo("UTC"))
            session.commit()

            print(
                f"[worker] ✅ team={team_id} attempt={row.attempt} "
                f"public(full={pub_full:.6f}, roi={pub_roi:.6f} n={pub_roi_n}) "
                f"latent_dim={row.latent_dim} size={size_mb:.2f}MB"
            )
            return pub_full
    except Exception as e:
        print(f"[worker] Fatal error while evaluating team {team_id}: {e}")
        return {"error": str(e)}
