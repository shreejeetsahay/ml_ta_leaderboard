import os, csv, json
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config

DATA_ROOT  = Path("data")
CACHE_ROOT = Path(config.CACHE_DIR)
IMG_SIZE   = config.IMG_SIZE
MIN_BOX    = 8
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"}

def step(m): print(f"[STEP] {m}")
def info(m): print(f"  - {m}")
def banner(m):
    print("\n" + "="*80); print(m); print("="*80)

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def fullframe_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

@lru_cache(maxsize=4096)
def _glob_by_name(name: str, root: Path) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None

def resolve_img_path(fn: str, csv_path: Path, root: Path) -> Path | None:
    p = Path(fn.strip().replace("\\", "/"))
    if str(p).startswith("./"): p = Path(str(p)[2:])
    cands = [
        csv_path.parent / p,
        csv_path.parent.parent / p,
        root / p,
        root / p.name,
        csv_path.parent / p.name,
    ]
    for c in cands:
        try:
            if c.exists(): return c.resolve()
        except Exception:
            pass
    hit = _glob_by_name(p.name, root)
    return hit.resolve() if hit else None

def parse_boxes_csv(csv_path: Path, data_root: Path, min_box=MIN_BOX):
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            try:
                fn_key = None
                for k in r.keys():
                    if k.strip().lstrip("\ufeff").lower() == "filename":
                        fn_key = k; break
                if fn_key is None: fn_key = list(r.keys())[0]
                fn = r[fn_key]
                x1 = int(float(r["Upper left corner X"]))
                y1 = int(float(r["Upper left corner Y"]))
                x2 = int(float(r["Lower right corner X"]))
                y2 = int(float(r["Lower right corner Y"]))
                if min(x2-x1, y2-y1) < min_box:
                    continue
                ip = resolve_img_path(fn, csv_path, data_root)
                if not ip or not ip.exists():
                    continue
                out.append((ip, x1, y1, x2, y2))
            except Exception:
                continue
    return out

def load_boxes_index(data_root: Path, include=("dayTrain", "daySequence1", "daySequence2")):
    ann_root = data_root / "Annotations"
    if not ann_root.exists():
        info("No Annotations/ found, skipping ROI cache.")
        return {}
    csvs = [p for p in ann_root.rglob("frameAnnotationsBOX.csv")
            if any(k in str(p).lower() for k in [s.lower() for s in include])]
    boxes_by_image = defaultdict(list)
    kept = 0
    for csvp in sorted(csvs):
        out = parse_boxes_csv(csvp, data_root)
        for (ip, x1, y1, x2, y2) in out:
            boxes_by_image[str(ip)].append((x1,y1,x2,y2))
        kept += len(out)
    info(f"Annotations kept: {kept} boxes on {len(boxes_by_image)} images")
    return boxes_by_image

def scale_box_to_resized(x1, y1, x2, y2, w0, h0, w, h):
    sx, sy = w / float(w0), h / float(h0)
    X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
    X2, Y2 = int(round(x2 * sx)), int(round(y2 * sy))
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(w, X2), min(h, Y2)
    if X2 <= X1 or Y2 <= Y1: return None
    return (X1, Y1, X2, Y2)

def mask_from_boxes(h, w, boxes):
    M = torch.zeros((1, h, w), dtype=torch.float32)  # (1,H,W)
    for b in boxes:
        if b is None: continue
        x1, y1, x2, y2 = b
        M[:, y1:y2, x1:x2] = 1.0
    return M

def write_shards(name: str, tensors: list[torch.Tensor], paths: list[str], outdir: Path, shard_size=2000):
    outdir.mkdir(parents=True, exist_ok=True)
    index = []
    n = len(tensors)
    s = 0
    k = 0
    while s < n:
        e = min(s + shard_size, n)
        shard_t = torch.stack(tensors[s:e])  # (B,C,H,W) or (B,1,H,W)
        shard_p = paths[s:e]
        shard_file = outdir / f"{name}_shard_{k:03d}.pt"
        torch.save({"tensors": shard_t}, shard_file)
        index.extend([{"path": p, "shard": shard_file.name, "offset": i} for i, p in enumerate(shard_p)])
        info(f"  wrote {shard_file}  [{s}:{e})")
        s = e; k += 1
    with open(outdir / f"{name}_index.json", "w") as f:
        json.dump(index, f)
    info(f"  wrote index: {outdir / f'{name}_index.json'}")

def build_public_cache_dayTrain():
    banner("Precompute (public) — dayTrain 256×256 tensors & ROI masks")
    tf = fullframe_transform()

    day_dir = DATA_ROOT / "dayTrain"
    if not day_dir.exists():
        raise FileNotFoundError(f"Missing {day_dir}")
    all_imgs = [p for p in day_dir.rglob("*") if is_image_file(p)]
    all_imgs.sort()
    info(f"dayTrain images: {len(all_imgs)}")

    boxes_by_image = load_boxes_index(DATA_ROOT, include=("dayTrain",))
    roi_set = set(boxes_by_image.keys())

    xs, x_paths = [], []
    for i, p in enumerate(all_imgs, 1):
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = tf(im)  # (3,256,256)
        xs.append(x)
        x_paths.append(str(p.resolve()))
        if i % 500 == 0: info(f"  resized {i}/{len(all_imgs)}")

    Ms, m_paths = [], []
    for i, p in enumerate(all_imgs, 1):
        sp = str(p.resolve())
        if sp not in roi_set:
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            w0, h0 = im.width, im.height
        H = W = IMG_SIZE
        scaled = [scale_box_to_resized(*b, w0, h0, W, H) for b in boxes_by_image[sp]]
        scaled = [b for b in scaled if b is not None]
        M = mask_from_boxes(H, W, scaled)  # (1,256,256)
        Ms.append(M)
        m_paths.append(sp)
        if i % 500 == 0: info(f"  mask {i}/{len(all_imgs)}")

    out = CACHE_ROOT / "public_dayTrain"
    write_shards("images", xs, x_paths, out / "images")
    write_shards("masks",  Ms, m_paths, out / "masks")
    info("Public cache ready.")

def build_private_cache_daySeq():
    banner("Precompute (private) — daySequence1/2 256×256 tensors & ROI masks")
    tf = fullframe_transform()
    boxes_by_image = load_boxes_index(DATA_ROOT, include=("daySequence1","daySequence2"))
    if not boxes_by_image:
        info("No private annotations found; skipping.")
        return

    priv_paths = sorted([Path(p) for p in boxes_by_image.keys() if Path(p).exists()])

    xs, x_paths, Ms, m_paths = [], [], [], []
    for i, p in enumerate(priv_paths, 1):
        with Image.open(p) as im:
            im = im.convert("RGB")
            w0, h0 = im.width, im.height
            x = tf(im)  
        H = W = IMG_SIZE
        scaled = [scale_box_to_resized(*b, w0, h0, W, H) for b in boxes_by_image[str(p)]]
        scaled = [b for b in scaled if b is not None]
        M = mask_from_boxes(H, W, scaled)  

        xs.append(x); x_paths.append(str(p.resolve()))
        Ms.append(M); m_paths.append(str(p.resolve()))

        if i % 500 == 0: info(f"  processed {i}/{len(priv_paths)}")

    out = CACHE_ROOT / "private_daySeq"
    write_shards("images", xs, x_paths, out / "images")
    write_shards("masks",  Ms, m_paths, out / "masks")
    info("Private cache ready.")


import json
from typing import Optional, Dict, Any

class _ShardBackedDataset(torch.utils.data.Dataset):
    """
    Generic shard-backed dataset for images-only OR masks-only.
    It reads <name>_index.json to map paths -> (shard_file, offset).
    For images: returns (x, path)  where x is (C,H,W) float32 [0,1].
    For masks : returns (M, path)  where M is (1,H,W) float32 {0,1}.
    """
    def __init__(self, root: Path, kind: str):
        """
        kind: "images" or "masks"
        """
        self.root = Path(root)
        self.kind = kind
        self.index_path = self.root / f"{kind}_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing index: {self.index_path}")
        with open(self.index_path, "r") as f:
            self.index = json.load(f) 

        self._items = [(rec["path"], rec["shard"], rec["offset"]) for rec in self.index]

        self._cur_shard_name = None
        self._cur_shard_tensors = None  

    def __len__(self):
        return len(self._items)

    def _ensure_shard_loaded(self, shard_name: str):
        if self._cur_shard_name == shard_name and self._cur_shard_tensors is not None:
            return
        shard_file = self.root / shard_name
        bundle = torch.load(shard_file, map_location="cpu")
        self._cur_shard_tensors = bundle["tensors"]  # (B,C,H,W) or (B,1,H,W)
        self._cur_shard_name = shard_name

    def __getitem__(self, i):
        path, shard_name, offset = self._items[i]
        self._ensure_shard_loaded(shard_name)
        t = self._cur_shard_tensors[offset]  # (C,H,W) or (1,H,W)
        return t, path


class _ROIDataset(torch.utils.data.Dataset):
    """
    Zips images and masks by path to yield (x, M, path).
    Only includes paths present in BOTH image index and mask index.
    """
    def __init__(self, images_root: Path, masks_root: Path):
        self.img_ds = _ShardBackedDataset(images_root, "images")
        self.msk_ds = _ShardBackedDataset(masks_root, "masks")

        # Build path -> (shard, offset) maps
        def _build_map(ds: _ShardBackedDataset):
            mp = {}
            for rec in ds.index:
                mp[rec["path"]] = (rec["shard"], rec["offset"])
            return mp

        self._img_map = _build_map(self.img_ds)
        self._msk_map = _build_map(self.msk_ds)

        # Eligible paths = intersection
        self.paths = sorted(set(self._img_map.keys()) & set(self._msk_map.keys()))

        # Local shard caches (separate from inner datasets)
        self._img_cur_name = None
        self._img_cur_tensors = None
        self._msk_cur_name = None
        self._msk_cur_tensors = None

    def __len__(self):
        return len(self.paths)

    def _ensure_loaded(self, kind: str, shard_name: str):
        if kind == "img":
            if self._img_cur_name == shard_name and self._img_cur_tensors is not None:
                return
            bundle = torch.load(self.img_ds.root / shard_name, map_location="cpu")
            self._img_cur_tensors = bundle["tensors"]
            self._img_cur_name = shard_name
        else:
            if self._msk_cur_name == shard_name and self._msk_cur_tensors is not None:
                return
            bundle = torch.load(self.msk_ds.root / shard_name, map_location="cpu")
            self._msk_cur_tensors = bundle["tensors"]
            self._msk_cur_name = shard_name

    def __getitem__(self, i):
        p = self.paths[i]
        img_shard, img_off = self._img_map[p]
        msk_shard, msk_off = self._msk_map[p]

        self._ensure_loaded("img", img_shard)
        self._ensure_loaded("msk", msk_shard)

        x = self._img_cur_tensors[img_off]  # (3,256,256)
        M = self._msk_cur_tensors[msk_off]  # (1,256,256)
        return x, M, p


def _build_full_loader(split_root: Path, batch_size: int, device: torch.device) -> torch.utils.data.DataLoader:
    """
    DataLoader that yields only images (or (images, paths)) for Full-MSE.
    We’ll yield just images to keep the worker simple.
    """
    base_ds = _ShardBackedDataset(split_root / "images", "images")

    class _ImagesOnlyDS(torch.utils.data.Dataset):
        def __len__(self): return len(base_ds)
        def __getitem__(self, i):
            x, _ = base_ds[i]        # x:(C,H,W)
            return x                 # return only tensor

    ds = _ImagesOnlyDS()
    num_workers = 4 if device.type == "cuda" else 0
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def load_cached_public(cache_dir: str, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """
    Returns:
      {
        "full_loader": DataLoader over all dayTrain images (pre-resized),
        "roi_dataset": Dataset over dayTrain annotated subset yielding (x, M, path)  OR None if no masks
      }
    """
    root = Path(cache_dir) / "public_dayTrain"
    images_dir = root / "images"
    masks_dir  = root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing public images cache at {images_dir}")

    out = {
        "full_loader": _build_full_loader(root, batch_size, device),
        "roi_dataset": None
    }
    if (masks_dir / "masks_index.json").exists():
        out["roi_dataset"] = _ROIDataset(images_dir, masks_dir)
    return out


def load_cached_private(cache_dir: str, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """
    Same structure as public, but for daySequence1+2.
    """
    root = Path(cache_dir) / "private_daySeq"
    images_dir = root / "images"
    masks_dir  = root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing private images cache at {images_dir}")

    out = {
        "full_loader": _build_full_loader(root, batch_size, device),
        "roi_dataset": None
    }
    if (masks_dir / "masks_index.json").exists():
        out["roi_dataset"] = _ROIDataset(images_dir, masks_dir)
    return out


def main():
    build_public_cache_dayTrain()
    build_private_cache_daySeq()
    banner("Precompute completed.")

if __name__ == "__main__":
    main()
