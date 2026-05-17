# CS 4774 Homework 2 Leaderboard

Flask web application for the **Traffic Light Autoencoder** competition (CS 4774 Machine Learning, University of Virginia). Teams register, submit TorchScript autoencoder models, and are ranked on reconstruction quality and compression metrics evaluated on a held-out traffic-light dataset.

## Overview

The server accepts scripted PyTorch models (`.pt` / `.pth`), evaluates them asynchronously on preprocessed image tensors, and displays results on public and admin leaderboards. Evaluation uses two metrics on held-out image splits (see **Dataset** below):

- **Full MSE** — mean squared error over the entire 256×256 image
- **ROI MSE** — mean squared error inside bounding-box regions from `frameAnnotationsBOX.csv` under `data/Annotations/`

Ranking also considers **latent dimension** (lower is better) and **model file size** (≤ 23 MB).

## Dataset

This project uses the **[LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)** (Kaggle: `mbornoe/lisa-traffic-light-dataset`), originally released by Aalborg University for traffic-light detection research.

| Source | Link |
|--------|------|
| Kaggle | https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset |
| Institutional record | https://vbn.aau.dk/en/datasets/lisa-traffic-light-dataset/ |

**Layout used by this leaderboard**

Images and annotations are separate top-level folders in `data/`:

| Role | Folder | Contents |
|------|--------|----------|
| Images (public) | `dayTrain/` | Frames scored on the public leaderboard |
| Images (private) | `daySequence1/`, `daySequence2/` | Optional holdout frames (admin view) |
| Annotations (both) | `Annotations/` | Bounding boxes for ROI-MSE on **public and private** splits |

Under `Annotations/` (often nested as `Annotations/Annotations/`), LISA places `frameAnnotationsBOX.csv` per clip or sequence—for example:

```
data/Annotations/Annotations/
├── dayTrain/dayClip1/frameAnnotationsBOX.csv
├── dayTrain/dayClip2/frameAnnotationsBOX.csv
├── …
├── daySequence1/frameAnnotationsBOX.csv
└── daySequence2/frameAnnotationsBOX.csv
```

`precompute_cache.py` and `datasets.py` read those CSVs to build ROI masks; full-frame MSE uses all images under the image folders above. Images are daytime frames from a roof-mounted stereo camera; this homework uses the **left-camera view** only.

**Citations**

```bibtex
@article{jensen2016vision,
  title={Vision for looking at traffic lights: Issues, survey, and perspectives},
  author={Jensen, Morten Born{\o} and Philipsen, Mark Philip and M{\o}gelmose, Andreas and Moeslund, Thomas Baltzer and Trivedi, Mohan Manubhai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={7},
  pages={1800--1815},
  year={2016},
  doi={10.1109/TITS.2015.2509509},
  publisher={IEEE}
}

@inproceedings{philipsen2015traffic,
  title={Traffic light detection: A learning algorithm and evaluations on challenging dataset},
  author={Philipsen, Mark Philip and Jensen, Morten Born{\o} and M{\o}gelmose, Andreas and Moeslund, Thomas B and Trivedi, Mohan M},
  booktitle={intelligent transportation systems (ITSC), 2015 IEEE 18th international conference on},
  pages={2341--2345},
  year={2015},
  organization={IEEE}
}
```

**Download into `data/`**

Download the dataset from Kaggle and extract it under the project’s `data/` folder (same level as `app.py`):

```bash
cd leaderboard
# Requires ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api)
kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data --unzip
```

After extraction, `data/` must include the image folders (`dayTrain/`, `daySequence1/`, `daySequence2/`) and the `Annotations/` tree with `frameAnnotationsBOX.csv` files for each split. See [Setup → Prepare data](#2-prepare-data) for the full layout.

Students may alternatively use the course bundle from `GET /download/train-dataset-hw2` and unzip it into `data/`.

## Architecture

```
┌─────────────┐     POST /submit      ┌──────────────┐
│   Client    │ ───────────────────►  │   app.py     │
│ (notebook)  │                       │  (Flask)     │
└─────────────┘                       └──────┬───────┘
                                             │ enqueue
                                             ▼
                                      ┌──────────────┐
                                      │  worker.py   │
                                      │ (background) │
                                      └──────┬───────┘
                                             │ load cache
                                             ▼
                                      ┌──────────────┐
                                      │ cache/       │
                                      │ (precomputed)│
                                      └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │leaderboard.db │
                                      │  (SQLite)    │
                                      └──────────────┘
```

| Component | Role |
|-----------|------|
| `app.py` | HTTP routes, registration, leaderboards, downloads |
| `worker.py` | Background thread; loads TorchScript, runs metrics, updates DB |
| `precompute_cache.py` | Resizes images to 256×256 and writes tensor shards to `cache/` |
| `models.py` | SQLAlchemy schema (`Team`, `Result`) and SQLite setup |
| `datasets.py` | Raw dataset loaders (used when building cache) |
| `config.py` | Paths, limits, scoring caps, device selection |

## Project structure

```
leaderboard/
├── app.py                 # Flask application entry point
├── worker.py              # Async model evaluation worker
├── precompute_cache.py    # Build evaluation cache (run once before serving)
├── config.py              # Configuration
├── models.py              # Database models
├── datasets.py            # Image / ROI dataset utilities
├── requirements.txt       # Python dependencies
├── templates/             # HTML pages (register, leaderboard, instructions)
├── static/download/       # starter.ipynb, training_dataset.zip
├── data/                  # Raw dataset (not in repo; see Setup)
│   ├── dayTrain/          # Public evaluation images
│   ├── daySequence1/      # Private holdout images (optional)
│   ├── daySequence2/
│   └── Annotations/       # ROI boxes (frameAnnotationsBOX.csv) for all splits
├── cache/                 # Precomputed tensors (generated)
├── submissions/           # Uploaded model files (generated)
└── leaderboard.db         # SQLite database (generated)
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (evaluation uses `cuda:0` when available; see `config.DEVICE`)
- [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) downloaded and extracted under `data/`

## Setup

### 1. Install dependencies

```bash
cd leaderboard
pip install -r requirements.txt
```

Install PyTorch separately if needed for your CUDA version ([pytorch.org](https://pytorch.org)).

### 2. Prepare data

Download the [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) from Kaggle and extract it into the **`data/`** folder at the project root:

```bash
kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data --unzip
```

Expected layout after extraction:

```
data/
├── dayTrain/                    # public evaluation images
├── daySequence1/                # private holdout images (optional)
├── daySequence2/
└── Annotations/
    └── Annotations/             # nested folder is normal for LISA
        ├── dayTrain/.../frameAnnotationsBOX.csv
        ├── daySequence1/frameAnnotationsBOX.csv
        └── daySequence2/frameAnnotationsBOX.csv
```

Also, to provide training dataset to students, zip the folders inside `dayTrain/dayTrain` as training_dataset.zip, and put it inside `static/download`.
### 3. Build the evaluation cache

Preprocessing resizes all images to 256×256 and writes sharded `.pt` files. This step is required before the worker can score submissions.

```bash
python precompute_cache.py
```

This creates:

- `cache/public_dayTrain/` — `dayTrain/` images + ROI masks from `Annotations/.../dayTrain/`
- `cache/private_daySeq/` — `daySequence1/` + `daySequence2/` images + ROI masks from `Annotations/.../daySequence*`

### 4. Configure environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_PASSWORD` | `MySecretPassword` | Password for the private admin leaderboard |

```bash
export ADMIN_PASSWORD="your-secure-password"
```

Other settings live in `config.py`: model size limit (23 MB), submission rate limit (15 min), minimum latent dim (8), batch size, score caps.

### 5. Run the server

```bash
python app.py
```

The app listens on `0.0.0.0:9000` by default.

> **Note:** The `/submit` route is currently commented out in `app.py`. To accept student uploads, uncomment the `submit()` handler (lines ~167–227) and ensure the background worker in `worker.py` is started (it starts automatically on import).

## Web routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Homework landing page |
| `/register-page` | GET | Team registration UI |
| `/instructions-hw2` | GET | Submission instructions |
| `/register-web` | POST | Register team (`team_name`, `computing_id`) → returns `token` |
| `/retrieve-token` | POST | Recover token by `computing_id` |
| `/submit` | POST | Upload model (`token`, `file`) — **must be enabled in code** |
| `/submission-status/<token>` | GET | List all submission attempts and statuses |
| `/leaderboard-hw2-final-1` | GET | Public leaderboard (latest successful submission per team) |
| `/leaderboard-hw2-private-view` | GET | Admin leaderboard (`?password=...`, optional `?download=csv`) |
| `/download/starter-hw2` | GET | Starter Jupyter notebook |
| `/download/train-dataset-hw2` | GET | Training dataset zip |

## API examples

**Register a team:**

```bash
curl -X POST http://localhost:9000/register-web \
  -H "Content-Type: application/json" \
  -d '{"team_name": "My Team", "computing_id": "abc1de"}'
```

**Submit a model** (when `/submit` is enabled):

```python
import requests

def submit_model(token: str, model_path: str, server_url="http://localhost:9000"):
    with open(model_path, "rb") as f:
        response = requests.post(
            f"{server_url}/submit",
            data={"token": token},
            files={"file": f},
        )
    return response.json()
```

**Check submission status:**

```bash
curl http://localhost:9000/submission-status/<YOUR_TOKEN>
```

## Scoring and ranking

For each successful submission, the worker records:

| Field | Description |
|-------|-------------|
| `latent_dim` | Bottleneck size (inferred from `enc(x).shape[1]`) |
| `public_full_mse` | Full-frame reconstruction MSE on `dayTrain/` images |
| `public_roi_mse` | ROI-masked MSE using `Annotations/.../dayTrain/` box CSVs |
| `model_size` | Uploaded file size in MB |

**Composite score** (display and optional ranking via `config.RANK_BY = "composite"`):

| Metric | Weight | Direction |
|--------|--------|-----------|
| Latent dimension | 40% | Lower is better |
| Full MSE | 35% | Lower is better |
| ROI MSE | 20% | Lower is better |
| Model size | 5% | Lower is better |

Metrics are min–max normalized using fixed caps in `config.SCORE_CAPS` (or dynamic cohort min/max if `USE_DYNAMIC_NORMALIZATION = True`).

**Default tie-break order** (when `RANK_BY = "tie"`): latent dim → full MSE → ROI MSE → model size → submission time.

## Model requirements

Submitted models must:

1. Be **TorchScript** files (`.pt` / `.pth`) from `torch.jit.script(model)`
2. Expose callable **`enc`** and **`dec`** methods (grading calls `dec(enc(x))`, not `model(x)`)
3. Return latent `z` with shape **`(B, latent_dim)`** (2D tensor)
4. Accept RGB input **`[B, 3, 256, 256]`** in `[0, 1]` and output the same shape
5. Have **`latent_dim ≥ 8`**
6. Be **≤ 23 MB** on disk

### Evaluation statuses

| Status | Meaning |
|--------|---------|
| `pending` | Queued or running |
| `successful` | Metrics computed; shown on leaderboard |
| `broken file` | TorchScript load failed |
| `rejected (file too large)` | Over 23 MB |
| `rejected (latent_dim < 8)` | Bottleneck too small |
| `rejected (zero public full_mse)` | Trivial zero-error solution |
| Other `*_eval failed` | Runtime error during metric computation |

## Utility scripts

| Script | Purpose |
|--------|---------|
| `precompute_cache.py` | Build or rebuild `cache/` from raw `data/` |
| `sanity.py` | Local evaluation of a model against cached tensors |
| `cudal.py` | Print PyTorch / CUDA availability |
| `sqlitequery.py` | Ad-hoc read-only queries against `leaderboard.db` |
| `submit1.py` | Example client submission script |

## Database

SQLite database at `leaderboard.db`:

- **`teams`** — `name`, `token`, `computing_id`, rate-limit timestamps
- **`results`** — one row per submission attempt with metrics and `status`

Schema migrations for new columns are applied automatically in `models.ensure_schema()`.

## Production notes

- Run behind a reverse proxy or process manager (e.g. `gunicorn`, `systemd`, `nohup`) for long-lived deployment.
- Rebuild `cache/` if the dataset or `IMG_SIZE` in `config.py` changes.
- The evaluation worker is a single daemon thread; heavy concurrent load may require a proper task queue.
- Do not commit secrets (`kaggle.json`, tokens in `submit1.py`) or large artifacts (`leaderboard.db`, `cache/`, `submissions/`).

## Course context

Homework 2 — **Traffic Light Autoencoder** (CS 4774, UVA). Students train convolutional autoencoders on traffic-light imagery, export TorchScript models, and compete on the public leaderboard while instructors can export full results from the password-protected admin view.
