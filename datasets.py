# Loads full images and builds ROI masks from frameAnnotationsBOX.csv
import os, csv
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def default_transform(img_size: int = 256, grayscale: bool = False):
    t = [transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)]
    if grayscale:
        t.append(transforms.Grayscale(1))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def _find_annotations_root(data_root: Path) -> Optional[Path]:
    cand1 = data_root / "Annotations"
    cand2 = data_root / "Annotations" / "Annotations"
    if cand2.exists(): return cand2
    if cand1.exists(): return cand1
    return None

@lru_cache(maxsize=4096)
def _glob_by_name_cached(name: str, data_root: Path) -> Optional[Path]:
    hits = list(data_root.rglob(name))
    return hits[0] if hits else None

def _resolve_img_path(fn: str, csv_path: Path, data_root: Path) -> Optional[Path]:
    p = Path(fn.strip().replace("\\", "/"))
    if str(p).startswith("./"):
        p = Path(str(p)[2:])
    candidates = [
        csv_path.parent / p,
        csv_path.parent.parent / p,
        data_root / p,
        data_root / p.name,
        csv_path.parent / p.name,
    ]
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            pass
    hit = _glob_by_name_cached(p.name, data_root)
    return hit.resolve() if hit else None

def _parse_boxes_csv(csv_path: Path, data_root: Path, min_box: int = 8):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            try:
                fn_key = None
                for k in r.keys():
                    if k.strip().lstrip("\ufeff").lower() == "filename":
                        fn_key = k; break
                if fn_key is None:
                    fn_key = list(r.keys())[0]
                fn = r[fn_key]
                x1 = int(float(r["Upper left corner X"]))
                y1 = int(float(r["Upper left corner Y"]))
                x2 = int(float(r["Lower right corner X"]))
                y2 = int(float(r["Lower right corner Y"]))
                if min(x2-x1, y2-y1) < min_box:
                    continue
                img_path = _resolve_img_path(fn, csv_path, data_root)
                if img_path and img_path.exists():
                    rows.append((img_path, x1, y1, x2, y2))
            except Exception:
                continue
    return rows

def load_boxes_index(data_root: Path, subsets=("daySequence1", "daySequence2", "dayTrain"), min_box=8) -> Dict[str, List[Tuple[int,int,int,int]]]:
    ann_root = _find_annotations_root(data_root)
    out: Dict[str, List[Tuple[int,int,int,int]]] = {}
    if not ann_root: return out
    for csv_path in ann_root.rglob("frameAnnotationsBOX.csv"):
        sp = str(csv_path).lower()
        if not any(s.lower() in sp for s in [s.lower() for s in subsets]):
            continue
        for (ip, x1, y1, x2, y2) in _parse_boxes_csv(csv_path, data_root, min_box):
            out.setdefault(str(ip), []).append((x1, y1, x2, y2))
    return out

def scale_box_to_resized(x1, y1, x2, y2, w0, h0, w, h):
    sx, sy = w / float(w0), h / float(h0)
    X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
    X2, Y2 = int(round(x2 * sx)), int(round(y2 * sy))
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(w, X2), min(h, Y2)
    if X2 <= X1 or Y2 <= Y1:
        return None
    return (X1, Y1, X2, Y2)

def mask_from_boxes(h: int, w: int, boxes, device=None):
    M = torch.zeros((1, h, w), device=device, dtype=torch.float32)
    for b in boxes:
        if b is None: continue
        x1, y1, x2, y2 = b
        M[:, y1:y2, x1:x2] = 1.0
    return M

class PlainImageDataset(Dataset):
    """Loads all images under data/<subset>/ (full frames). Returns (x, 0, path)."""
    def __init__(self, root_dir: Path, subset: str, transform=None, grayscale=False, img_size=256):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.dir = self.root_dir / subset
        if transform is None:
            transform = default_transform(img_size, grayscale)
        self.transform = transform
        if not self.dir.exists():
            raise FileNotFoundError(f"Subset not found: {self.dir}")
        self.paths = [str(p.resolve()) for p in self.dir.rglob("*") if is_image_file(p)]
        self.paths.sort()
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, 0, p

class DaySequenceEvalDataset(Dataset):
    """Yields (x, M, path) with ROI mask from BOX CSVs."""
    def __init__(self, root_dir: Path, subsets=("daySequence1", "daySequence2"), transform=None, grayscale=False, img_size=256, min_box=8):
        self.root_dir = Path(root_dir)
        if transform is None:
            transform = default_transform(img_size, grayscale)
        self.transform = transform
        self.boxes_by_image = load_boxes_index(self.root_dir, subsets, min_box=min_box)
        self.paths = [p for p in self.boxes_by_image.keys() if Path(p).exists()]
        self.paths.sort()
        self.img_size = img_size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            w0, h0 = im.width, im.height
            x = self.transform(im)
        _, H, W = x.shape
        scaled = [scale_box_to_resized(x1, y1, x2, y2, w0, h0, W, H) for (x1, y1, x2, y2) in self.boxes_by_image[p]]
        scaled = [b for b in scaled if b is not None]
        M = mask_from_boxes(H, W, scaled)
        return x, M, p
