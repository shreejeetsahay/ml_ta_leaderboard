from flask import Flask, request, jsonify, render_template, send_file, Response, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import os, secrets, io
import pandas as pd
from pathlib import Path
import csv
import config
from models import SessionLocal, Team, Result
from worker import submission_queue
# app.py (top-level, after imports)
from precompute_cache import main as build_cache_main


# --------- Scoring helpers ---------
def _cap_range(vals, k):
    # fall back to fixed caps if dynamic disabled or values empty
    caps = config.SCORE_CAPS[k]
    vmin, vmax = caps["min"], caps["max"]
    if config.USE_DYNAMIC_NORMALIZATION and vals:
        vmin = min(vals); vmax = max(vals)
        if vmin == vmax:
            # avoid divide-by-zero; widen trivially
            vmax = vmin + 1e-12
    return vmin, vmax

def _norm_inverse(x, vmin, vmax):
    # lower-is-better -> normalized high-is-good in [0,1]
    if x is None:
        return None
    # clamp
    x = max(vmin, min(vmax, x))
    return 1.0 - (x - vmin) / (vmax - vmin)

def compute_composite(rows, public=True):
    """
    rows: list of dicts already shaped for template.
    Adds 'weighted_score' ∈ [0,1] (None if any metric missing).
    Uses public_* metrics on public board, private_* on private board.
    """
    # pull metric lists (ignore None)
    if public:
        full_key = "public_full_mse"; roi_key = "public_roi_mse"; computing_id_key = "computing_id"
    else:
        full_key = "Public Full MSE"; roi_key = "Public ROI MSE"; computing_id_key = "Computing ID"  # note: private route builds different keys

    ld_vals   = [r.get("latent_dim" if public else "Latent Dim") for r in rows if r.get("latent_dim" if public else "Latent Dim") is not None]
    fm_vals   = [r.get(full_key) for r in rows if r.get(full_key) is not None]
    roi_vals  = [r.get(roi_key)  for r in rows if r.get(roi_key)  is not None]
    sz_vals   = [r.get("model_size" if public else "Model Size (MB)") for r in rows if r.get("model_size" if public else "Model Size (MB)") is not None]

    ld_min, ld_max   = _cap_range(ld_vals,   "latent_dim")
    fm_min, fm_max   = _cap_range(fm_vals,   "full_mse")
    roi_min, roi_max = _cap_range(roi_vals,  "roi_mse")
    sz_min, sz_max   = _cap_range(sz_vals,   "model_size")

    for r in rows:
        # fetch per board
        ld  = r.get("latent_dim" if public else "Latent Dim")
        fm  = r.get(full_key)
        roi = r.get(roi_key)
        sz  = r.get("model_size" if public else "Model Size (MB)")

        # if any missing, skip scoring
        if any(v is None for v in (ld, fm, roi, sz)):
            r["weighted_score"] = None
            continue

        s_ld  = _norm_inverse(ld,  ld_min,  ld_max)
        s_fm  = _norm_inverse(fm,  fm_min,  fm_max)
        s_roi = _norm_inverse(roi, roi_min, roi_max)
        s_sz  = _norm_inverse(sz,  sz_min,  sz_max)

        # weights: LD 40, Full MSE 35, ROI 20, Size 5
        r["weighted_score"] = 0.40*s_ld + 0.35*s_fm + 0.20*s_roi + 0.05*s_sz
        # if r[computing_id_key] in ["abs6bd", "twz4wq", "rfb3mg", "psw2uw", "svb9ux", "nyx7ck", "nzq2uu", "eyu8ps"]:
        #     r["weighted_score"] -= 0.1

    return rows

def _none_last(x):
    return (x is None, x)

def sort_rows(rows, public=True):
    if config.RANK_BY == "composite":
      # High score first, then tie-breakers
      key = (lambda r: (
          _none_last(-r.get("weighted_score") if r.get("weighted_score") is not None else None),
          _none_last(r.get("latent_dim" if public else "Latent Dim")),
          _none_last(r.get("public_full_mse" if public else "Public Full MSE")),
          _none_last(r.get("public_roi_mse" if public else "Public ROI MSE")),
          _none_last(r.get("model_size" if public else "Model Size (MB)")),
          _none_last(r.get("submitted_at" if public else "Submitted At")),
      ))
      rows.sort(key=key)
      return rows

    # Default: tie-break rank only (LD ↓ → Full MSE ↓ → ROI ↓ → Size ↓ → time)
    rows.sort(key=lambda r: (
        _none_last(r.get("latent_dim" if public else "Latent Dim")),
        _none_last(r.get("public_full_mse" if public else "Public Full MSE")),
        _none_last(r.get("public_roi_mse" if public else "Public ROI MSE")),
        _none_last(r.get("model_size" if public else "Model Size (MB)")),
        _none_last(r.get("submitted_at" if public else "Submitted At")),
    ))
    return rows

# Build cache ONCE at process start (idempotent; shards won’t re-write if you add a guard).

app = Flask(__name__)
os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

# ---------- Landing ----------
@app.route("/", methods=["GET"])
def home():
    # your uploaded landing page
    return render_template("homework2.html")  # :contentReference[oaicite:3]{index=3}

# ---------- Registration UI ----------
@app.route("/register-page", methods=["GET"])
def register_page():
    return render_template("register.html")  # :contentReference[oaicite:4]{index=4}

# ---------- Instructions ----------
@app.route("/instructions-hw2", methods=["GET"])
def instructions_hw2():
    return render_template("instructions_hw2.html")  # :contentReference[oaicite:5]{index=5}

# ---------- API: Register via web (returns token) ----------
@app.route("/register-web", methods=["POST"])
def register_web():
    session = SessionLocal()
    try:
        name = request.json.get("team_name", "").strip()
        computing_id = request.json.get("computing_id", "").strip()
        if not name or not computing_id:
            return jsonify({"error": "team_name and computing_id required"}), 400
        if session.query(Team).filter_by(name=name).first():
            return jsonify({"error": "Team name already taken"}), 400
        if session.query(Team).filter_by(computing_id=computing_id).first():
            return jsonify({"error": "Computing ID already registered"}), 400
        token = secrets.token_hex(16)
        t = Team(name=name, token=token, computing_id=computing_id)
        session.add(t); session.commit()
        return jsonify({"message":"Registration successful","team_name":t.name,"computing_id":t.computing_id,"token":token}), 201
    finally:
        session.close()

# ---------- API: Retrieve token ----------
@app.route("/retrieve-token", methods=["POST"])
def retrieve_token():
    session = SessionLocal()
    try:
        computing_id = request.json.get("computing_id")
        if not computing_id:
            return jsonify({"error": "computing_id required"}), 400
        t = session.query(Team).filter_by(computing_id=computing_id).first()
        if not t:
            return jsonify({"error": "No team found with this computing ID"}), 404
        return jsonify({"token": t.token})
    finally:
        session.close()

# # ---------- API: Submit model (.pt/.pth TorchScript) ----------
# @app.route("/submit", methods=["POST"])
# def submit():
#     token = request.form.get("token")
#     file = request.files.get("file")
#     if not token or not file:
#         return jsonify({"error": "Missing token or file"}), 400

#     session = SessionLocal()
#     try:
#         team = session.query(Team).filter_by(token=token).first()
#         if not team:
#             return jsonify({"error": "Invalid token"}), 403

#         # rate limit 15 min
#         now_utc = datetime.now(tz=ZoneInfo("UTC"))
#         last_sub = team.last_submission
#         if last_sub and last_sub.tzinfo is None:
#             last_sub = last_sub.replace(tzinfo=ZoneInfo("UTC"))
#         if last_sub and (now_utc - last_sub) < timedelta(minutes=config.RATE_LIMIT_MINUTES):
#             remaining = config.RATE_LIMIT_MINUTES - int((now_utc - last_sub).total_seconds() // 60)
#             return jsonify({"error": f"Rate limit: wait {remaining} minutes"}), 429

#         filename = secure_filename(file.filename)
#         if not filename.lower().endswith((".pt", ".pth")):
#             return jsonify({"error": "Only .pt or .pth files are allowed"}), 400

#         # size limit 23 MB
#         file.seek(0, os.SEEK_END)
#         size_mb = file.tell() / (1024*1024)
#         file.seek(0)
#         if size_mb > config.MAX_MODEL_SIZE:
#             return jsonify({"error": f"File too large (>{config.MAX_MODEL_SIZE} MB)"}), 400

#         team.last_submission = now_utc

#         latest_model_path = Path(config.SUBMISSION_DIR) / f"{team.computing_id or team.name}_latest.pt"
#         file.save(latest_model_path.as_posix())

#         # attempt #
#         last_attempt = (session.query(Result)
#                         .filter_by(team_id=team.id)
#                         .order_by(Result.attempt.desc())
#                         .first())
#         next_attempt = last_attempt.attempt + 1 if last_attempt else 1
        
#         r = Result(team_id=team.id, score=None, attempt=next_attempt,
#                    submitted_at=now_utc, model_size=size_mb, status="pending", artifact=latest_model_path.name)
#         session.add(r); session.commit()
#         submission_queue.put((team.id, latest_model_path.as_posix()))

#         # Queue evaluation
        
#         return jsonify({
#             "message": f"Submission received for team '{team.name}'. Attempt #{next_attempt}.",
#             "status": "pending", "attempt": next_attempt
#         })
#     except Exception as e:
#         session.rollback()
#         return jsonify({"error": f"Failed to process submission: {e}"}), 500
#     finally:
#         session.close()

# ---------- API: Check submission status (all attempts) ----------
@app.route("/submission-status/<token>", methods=["GET"])
def submission_status_all(token):
    session = SessionLocal()
    try:
        team = session.query(Team).filter_by(token=token).first()
        if not team:
            return jsonify({"error": "Invalid token"}), 403

        # throttle status polling: 15s
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        last_check = team.last_status_check
        if last_check and last_check.tzinfo is None:
            last_check = last_check.replace(tzinfo=ZoneInfo("UTC"))
        if last_check and (now_utc - last_check) < timedelta(seconds=15):
            remaining = 15 - int((now_utc - last_check).total_seconds())
            return jsonify({"error": f"Rate limit: wait {remaining} seconds"}), 429

        team.last_status_check = now_utc
        session.commit()

        results = (session.query(Result)
                   .filter_by(team_id=team.id)
                   .order_by(Result.attempt)
                   .all())

        eastern = ZoneInfo("America/New_York")
        out = []
        for r in results:
            local_time = r.submitted_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(eastern) if r.submitted_at else None
            out.append({
                "attempt": r.attempt,
                "model_size": r.model_size,
                "submitted_at": local_time.strftime("%b %d, %Y %I:%M:%S %p") if local_time else "",
                "status": r.status
            })
        out.sort(key=lambda x: x["attempt"])
        return jsonify(out)
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"Failed to fetch submission status: {str(e)}"}), 500
    finally:
        session.close()

# ---------- Public Leaderboard (dayTrain) ----------
@app.route("/leaderboard-hw2-final-1", methods=["GET"])
def leaderboard_public():
    with SessionLocal() as session:
        subq = (session.query(Result.team_id, func.max(Result.submitted_at).label("latest"))
                .filter(Result.status=="successful")
                .group_by(Result.team_id).subquery())
        rows = (session.query(
                    Team.name.label("team_name"),
                    Team.computing_id.label("computing_id"),
                    Result.latent_dim, Result.public_full_mse, Result.public_roi_mse,
                    Result.model_size, Result.submitted_at
                )
                .join(subq, Team.id==subq.c.team_id)
                .join(Result, (Result.team_id==subq.c.team_id) & (Result.submitted_at==subq.c.latest))
                .filter(Result.status=="successful")
                .all())

        eastern = ZoneInfo("America/New_York")
        output = []
        for r in rows:
            submitted_str = r.submitted_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(eastern).strftime("%b %d, %Y %I:%M:%S %p") if r.submitted_at else ""
            output.append({
                "team_name": r.team_name,
                "computing_id": r.computing_id,
                "latent_dim": r.latent_dim,
                "public_full_mse": r.public_full_mse,
                "public_roi_mse": r.public_roi_mse,
                "model_size": r.model_size,
                "submitted_at": submitted_str,
            })

        def none_last(x):
            return (x is None, x)
        # output.sort(key=lambda x: (
        #     none_last(x["latent_dim"]),
        #     none_last(x["public_full_mse"]),
        #     none_last(x["public_roi_mse"]), 
        #     none_last(x["model_size"]),       ))
        output = compute_composite(output, public=True)
        output = sort_rows(output, public=True)
    return render_template("leaderboard.html", results=output, current_year=datetime.now().year)

# ---------- Private Leaderboard (admin; daySequence1+2) ----------
from functools import wraps
def requires_auth(f):
    @wraps(f)
    def dec(*args, **kwargs):
        pwd = request.args.get("password")
        if pwd != config.ADMIN_PASSWORD:
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return dec

@app.route("/leaderboard-hw2-private-view", methods=["GET"])
@requires_auth
def leaderboard_private():
    with SessionLocal() as session:
        subq = (session.query(Result.team_id, func.max(Result.submitted_at).label("latest"))
                .filter(Result.status=="successful")
                .group_by(Result.team_id).subquery())
        rows = (session.query(
                    Team.name.label("team_name"),
                    Team.computing_id.label("computing_id"),
                    Result.latent_dim,
                    Result.public_full_mse, Result.public_roi_mse,
                    Result.model_size, Result.submitted_at, Result.attempt
                )
                .join(subq, Team.id==subq.c.team_id)
                .join(Result, (Result.team_id==subq.c.team_id) & (Result.submitted_at==subq.c.latest))
                .filter(Result.status=="successful")
                .all())

        eastern = ZoneInfo("America/New_York")
        output = []
        for r in rows:
            submitted_dt = r.submitted_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(eastern) if r.submitted_at else None
            output.append({
                "Team Name": r.team_name,
                "Computing ID": r.computing_id,
                "Latent Dim": r.latent_dim,
                "Public Full MSE": r.public_full_mse,
                "Public ROI MSE": r.public_roi_mse,
                "Model Size (MB)": r.model_size,
                "Submitted At": submitted_dt.strftime("%b %d, %Y %I:%M:%S %p") if submitted_dt else "",
                "Attempt": r.attempt
            })

        def none_last(x): return (x is None, x)
        # output.sort(key=lambda x: (
        #     none_last(x["Latent Dim"]),
        #     none_last(x["Private Full MSE"]),
        #     none_last(x["Private ROI MSE"]),
        #     none_last(x["Model Size (MB)"]),
        #     none_last(x["Submitted At"]),
        # ))
        # compute weighted score for display
        output = compute_composite(output, public=False)

        # sort by requested policy (tie-breaks by default)
        output = sort_rows(output, public=False)
    # --- CSV download ---
    if request.args.get("download") == "csv":
        fieldnames = [
            "Team Name",
            "Computing ID",
            "weighted_score",
            "Latent Dim",
            "Public Full MSE",
            "Public ROI MSE",
            "Model Size (MB)",
            "Submitted At",
            "Attempt",
        ]
        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=fieldnames)
        writer.writeheader()

        for row in output:
            ws = row.get("weighted_score")
            # format weighted_score to 3 decimals, or empty if None
            if ws is not None:
                ws_str = f"{ws:.3f}"
            else:
                ws_str = ""

            writer.writerow({
                "Team Name":        row.get("Team Name", ""),
                "Computing ID":     row.get("Computing ID", ""),
                "weighted_score":   ws_str,
                "Latent Dim":       row.get("Latent Dim", ""),
                "Public Full MSE":  row.get("Public Full MSE", ""),
                "Public ROI MSE":   row.get("Public ROI MSE", ""),
                "Model Size (MB)":  row.get("Model Size (MB)", ""),
                "Submitted At":     row.get("Submitted At", ""),
                "Attempt":          row.get("Attempt", ""),
            })

        resp = Response(si.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = "attachment; filename=leaderboard_private.csv"
        return resp

    # Excel download (optional)
    if request.args.get("download") == "excel":
        df = pd.DataFrame(output)
        buf = io.BytesIO(); df.to_excel(buf, index=False); buf.seek(0)
        return send_file(buf, download_name="leaderboard_private.xlsx", as_attachment=True)

    return render_template("admin_leaderboard.html", results=output)

# @app.route("/leaderboard.csv")
# def leaderboard_csv():
#     # Same results you use for the HTML table
#     # Example structure:
#     # results = [
#     #   {"Team Name": "...", "Computing ID": "...", "weighted_score": 0.95, ...},
#     #   ...
#     # ]
#     output = io.StringIO()
#     writer = csv.writer(output)

#     # Header row (match your table columns)
#     header = [
#         "Team Name",
#         "Computing ID",
#         "Weighted Score",
#         "Latent Dim",
#         "Public Full MSE",
#         "Public ROI MSE",
#         "Model Size (MB)",
#         "Submitted At",
#     ]
#     writer.writerow(header)

#     for r in results:
#         writer.writerow([
#             r.get("Team Name", ""),
#             r.get("Computing ID", ""),
#             r.get("weighted_score", ""),
#             r.get("Latent Dim", ""),
#             r.get("Public Full MSE", ""),
#             r.get("Public ROI MSE", ""),
#             r.get("Model Size (MB)", ""),
#             r.get("Submitted At", ""),
#         ])

#     csv_data = output.getvalue()
#     output.close()

#     return Response(
#         csv_data,
#         mimetype="text/csv",
#         headers={"Content-Disposition": "attachment; filename=leaderboard.csv"}
#     )
# ---------- Downloads ----------
@app.route("/download/train-dataset-hw2")
def download_train_hw2():
    return send_from_directory("static/download", "training_dataset.zip", as_attachment=True)

@app.route("/download/starter-hw2")
def download_starter_hw2():
    return send_from_directory("static/download", "starter.ipynb", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
