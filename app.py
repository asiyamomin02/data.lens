from flask import Flask, render_template, request, jsonify, send_file, session
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import requests
import io
import time
import uuid
import numpy as np
import os
import json
import re
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()


class SafeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)


app = Flask(__name__, template_folder="templates")
app.json = SafeJSONProvider(app)
# Session cookie ke liye secret key zaroori hai — .env me SECRET_KEY set kar sakti ho,
# warna ek random key generate ho jayegi (server restart pe sab sessions invalid ho jayenge,
# demo ke liye chalega, production ke liye .env me fix key rakhna behtar hai)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
print("API_KEY loaded:", bool(API_KEY))

# ------------------------------------------------------------------
# PER-USER (session-based) STORAGE
# ------------------------------------------------------------------
# Pehle ye df_store, last_filtered, last_analysis global dicts the —
# matlab agar 2 log alag-alag files upload karte, dono ka data mix ho
# jaata. Ab har user ko ek unique session_id milta hai (Flask session
# cookie se), aur uska data alag rakha jaata hai is dictionary me:
#   USER_DATA[session_id] = {"df": ..., "filtered": ..., "filename": ..., "analysis_text": ...}
# ------------------------------------------------------------------
USER_DATA = {}


def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_user_store():
    sid = get_session_id()
    if sid not in USER_DATA:
        USER_DATA[sid] = {
            "df": None,
            "filtered": None,
            "filename": None,
            "analysis_text": ""
        }
    return USER_DATA[sid]


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def read_uploaded_file(file):
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        content = file.read().decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(content))

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        data = file.read()
        return pd.read_excel(io.BytesIO(data))

    raise ValueError("Please upload a CSV or Excel file (.csv, .xlsx, .xls)")


def try_parse_dates(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            date_hint = any(
                x in col.lower()
                for x in ["date", "time", "month", "year", "created", "updated", "start", "end", "joined", "dob"]
            )

            if date_hint:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() >= max(1, len(df) // 5):
                        df[col] = parsed
                except Exception:
                    pass
    return df


def call_ai(system, user, max_tokens=400, timeout_seconds=60, max_attempts=3):
    if not API_KEY:
        return "AI is disabled because API key is missing."

    if len(user) > 2800:
        user = user[:2800] + "\n...(truncated)"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    max_attempts = max(1, int(max_attempts))
    for attempt in range(max_attempts):
        try:
            r = requests.post(MODEL_URL, headers=headers, json=payload, timeout=timeout_seconds)

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            if r.status_code == 429:
                if attempt < max_attempts - 1:
                    time.sleep(4 + attempt * 3)
                    continue
                return "Rate limit reached. Wait a few seconds and retry."

            try:
                return f"Error: {r.json().get('error', {}).get('message', 'Unknown')}"
            except Exception:
                return f"Error: HTTP {r.status_code}"

        except requests.exceptions.Timeout:
            if attempt < max_attempts - 1:
                time.sleep(3)
                continue
            return "Error: AI took too long to respond. Please try again."
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to AI service. Check your internet connection."
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(3)
                continue
            return f"Error: {str(e)}"

    return "Rate limit reached. Wait a few seconds and retry."


# ------------------------------------------------------------------
# GENERIC (file-agnostic) smart_detect
# ------------------------------------------------------------------
def smart_detect(df):
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    date_hint_words = ["date", "month", "year", "time", "period", "created", "closed", "start", "end", "joined"]
    date_candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_candidates.append(col)
        elif any(w in col.lower() for w in date_hint_words):
            date_candidates.append(col)
    date_col = date_candidates[0] if date_candidates else None

    cat_info = []
    for c in cat:
        if c == date_col:
            continue
        try:
            nun = df[c].nunique(dropna=True)
        except Exception:
            continue
        if nun >= 1:
            cat_info.append((c, nun))

    cat_info.sort(key=lambda x: x[1])

    n_rows = max(1, len(df))
    cardinality_limit = min(60, max(2, int(n_rows * 0.5)))

    status_col = next((c for c, n in cat_info if 2 <= n <= 8), None)

    group_candidates = [c for c, n in cat_info if c != status_col and 2 <= n <= cardinality_limit]
    branch_col = group_candidates[0] if len(group_candidates) >= 1 else None
    emp_col = group_candidates[1] if len(group_candidates) >= 2 else None

    used = {status_col, branch_col, emp_col, date_col}
    remaining_cat = [c for c in cat if c not in used]
    name_col = remaining_cat[0] if remaining_cat else None

    def looks_like_id(col):
        lc = col.lower()
        if lc.endswith("id") or lc == "id":
            return True
        s = df[col].dropna()
        if len(s) > 1 and s.nunique() == len(s) and s.is_monotonic_increasing:
            return True
        return False

    value_candidates = [c for c in num if not looks_like_id(c)]
    value_col = None
    if value_candidates:
        try:
            value_col = max(
                value_candidates,
                key=lambda c: df[c].dropna().std() if df[c].dropna().std() == df[c].dropna().std() else 0
            )
        except Exception:
            value_col = value_candidates[0]

    def find_by_name(keywords, pool):
        return next((c for c in pool if any(k in c.lower() for k in keywords)), None)

    discount_col = find_by_name(["discount", "disc", "reduction", "rebate"], num)
    booking_col = find_by_name(["booking", "advance", "deposit", "payment"], num)

    return {
        "value_col": value_col,
        "discount_col": discount_col,
        "booking_col": booking_col,
        "branch_col": branch_col,
        "emp_col": emp_col,
        "status_col": status_col,
        "date_col": date_col,
        "name_col": name_col,
        "num_cols": num,
        "cat_cols": cat,
    }


def compute_kpis(df, det):
    kpis = {"total_records": int(len(df))}

    vc = det["value_col"]
    sc = det["status_col"]
    dc = det["discount_col"]
    bc = det["booking_col"]
    br = det["branch_col"]
    ec = det["emp_col"]

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            kpis.update({
                "total_value": round(float(s.sum()), 0),
                "avg_value": round(float(s.mean()), 0),
                "max_value": round(float(s.max()), 0),
                "min_value": round(float(s.min()), 0),
                "value_col": vc
            })

    if sc and sc in df.columns:
        vc_counts = df[sc].astype(str).value_counts()
        win_vals = [v for v in vc_counts.index if any(k in str(v).lower() for k in ["won", "win", "closed", "success", "complete"])]

        if win_vals and len(df) > 0:
            won = int(vc_counts[win_vals].sum())
            kpis.update({
                "win_rate": round((won / len(df)) * 100, 1),
                "won_count": won
            })

        kpis["status_breakdown"] = {str(k): int(v) for k, v in vc_counts.head(6).items()}
        kpis["status_col"] = sc

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            kpis.update({
                "avg_discount": round(float(s.mean()), 0),
                "total_discount": round(float(s.sum()), 0),
                "discount_col": dc
            })

    if bc and bc in df.columns:
        s = df[bc].dropna()
        if not s.empty:
            kpis.update({
                "avg_booking": round(float(s.mean()), 0),
                "booking_col": bc
            })

    if br and br in df.columns:
        kpis["branch_count"] = int(df[br].nunique())
        kpis["branch_col"] = br
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(br)[vc].sum()
                if not grp.empty:
                    kpis["top_branch"] = str(grp.idxmax())
            except Exception:
                pass

    if ec and ec in df.columns:
        kpis["emp_count"] = int(df[ec].nunique())
        kpis["emp_col"] = ec
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(ec)[vc].sum()
                if not grp.empty:
                    kpis["top_employee"] = str(grp.idxmax())
            except Exception:
                pass

    return kpis


def compute_charts(df, det):
    charts = {}

    vc = det["value_col"]
    br = det["branch_col"]
    ec = det["emp_col"]
    sc = det["status_col"]
    dc = det["discount_col"]
    dtc = det["date_col"]

    if br and vc and br in df.columns and vc in df.columns:
        try:
            g = df.groupby(br)[vc].sum().round(0).sort_values(ascending=False).head(10)
            if not g.empty:
                charts["branch_revenue"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [float(v) for v in g.values],
                    "title": f"{vc} by {br}"
                }

            g2 = df.groupby(br).size().sort_values(ascending=False).head(10)
            if not g2.empty:
                charts["branch_deals"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": f"Count by {br}"
                }
        except Exception:
            pass

    if ec and vc and ec in df.columns and vc in df.columns:
        try:
            g = df.groupby(ec)[vc].sum().round(0).sort_values(ascending=False).head(10)
            if not g.empty:
                charts["emp_revenue"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [float(v) for v in g.values],
                    "title": f"{vc} by {ec}"
                }

            g2 = df.groupby(ec).size().sort_values(ascending=False).head(10)
            if not g2.empty:
                charts["emp_deals"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": f"Count by {ec}"
                }
        except Exception:
            pass

    if sc and sc in df.columns:
        try:
            g = df[sc].astype(str).value_counts().head(8)
            if not g.empty:
                charts["status_pie"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [int(v) for v in g.values],
                    "title": f"Distribution of {sc}"
                }
        except Exception:
            pass

    if dtc and dtc in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[dtc], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]

            if vc and vc in df2.columns:
                g = df2.groupby("_m")[vc].sum().round(0).sort_index().tail(12)
                if not g.empty:
                    charts["monthly_revenue"] = {
                        "labels": list(g.index.astype(str)),
                        "values": [float(v) for v in g.values],
                        "title": f"Monthly {vc}"
                    }

            g2 = df2.groupby("_m").size().sort_index().tail(12)
            if not g2.empty:
                charts["monthly_count"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": "Monthly Count"
                }
        except Exception:
            pass

    if dc and vc and dc in df.columns and vc in df.columns:
        try:
            s = df[[dc, vc]].dropna().head(300)
            if not s.empty:
                charts["scatter"] = {
                    "x": [float(v) for v in s[dc].round(0).values],
                    "y": [float(v) for v in s[vc].round(0).values],
                    "title": f"{dc} vs {vc}",
                    "xlabel": dc,
                    "ylabel": vc
                }
        except Exception:
            pass

    if not charts:
        num_cols = det["num_cols"]
        cat_cols = det["cat_cols"]

        if num_cols:
            for col in num_cols[:3]:
                s = df[col].dropna()
                if not s.empty:
                    charts[f"hist_{col}"] = {
                        "labels": ["min", "mean", "max"],
                        "values": [float(s.min()), float(s.mean()), float(s.max())],
                        "title": f"Stats of {col}"
                    }

        elif cat_cols:
            for col in cat_cols[:2]:
                g = df[col].astype(str).value_counts().head(10)
                if not g.empty:
                    charts[f"cat_{col}"] = {
                        "labels": list(g.index.astype(str)),
                        "values": [int(v) for v in g.values],
                        "title": f"Top values in {col}"
                    }

    return charts


def auto_insights(df, det, kpis):
    ins = []

    vc = det["value_col"]
    br = det["branch_col"]
    ec = det["emp_col"]
    dc = det["discount_col"]
    dtc = det["date_col"]

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            ins.append(f"Total {vc}: {s.sum():,.0f}")
            ins.append(f"Average {vc}: {s.mean():,.0f}")
            ins.append(f"Highest {vc}: {s.max():,.0f}")

    if br and vc and br in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(br)[vc].sum()
            if not top.empty:
                ins.append(f"Top {br}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if ec and vc and ec in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(ec)[vc].sum()
            if not top.empty:
                ins.append(f"Top {ec}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if "win_rate" in kpis:
        ins.append(f"Win Rate: {kpis['win_rate']}% ({kpis['won_count']} won / {len(df)} total)")

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            ins.append(f"Avg {dc}: {s.mean():,.0f}")

    if dtc and dtc in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[dtc], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]
            if not df2.empty:
                best = df2.groupby("_m").size().idxmax()
                ins.append(f"Busiest Month: {best}")
        except Exception:
            pass

    miss = int(df.isnull().sum().sum())
    ins.append(f"Missing Values: {miss}" if miss else "Data Quality: Clean")

    return ins


def build_col_stats(df, num_cols):
    col_stats = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            col_stats[c] = {
                "mean": None,
                "min": None,
                "max": None,
                "median": None,
                "missing": int(df[c].isnull().sum()),
                "total": None
            }
        else:
            col_stats[c] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "median": round(float(s.median()), 2),
                "missing": int(df[c].isnull().sum()),
                "total": round(float(s.sum()), 0)
            }
    return col_stats


def build_filter_options(df, det):
    """Return only compact categorical fields that make useful filter controls."""
    filter_options = {}
    for col in det["cat_cols"]:
        try:
            unique_count = int(df[col].nunique(dropna=True))
            if 1 < unique_count <= 60:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                # Smart filters are intended for compact category values (such as
                # country or status), not long workbook instructions/notes. Without
                # this guard a short practice workbook can surface its full text as
                # an unusable, visually noisy dropdown.
                value_lengths = [len(value.strip()) for value in values]
                is_instruction_like = (
                    any("\n" in value for value in values)
                    or max(value_lengths, default=0) > 80
                    or (sum(value_lengths) / max(1, len(value_lengths))) > 42
                )
                if is_instruction_like:
                    continue
                filter_options[col] = values
        except Exception:
            pass
    return filter_options


DATE_HINTS = ("date", "time", "month", "year", "created", "updated", "start", "end", "joined", "dob")


def parse_dates_safely(values):
    """Prefer Pandas' mixed parser while retaining compatibility with older versions."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def _date_signature(value):
    text = str(value).strip()
    if re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text):
        return "year-first"
    if re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", text):
        return "day-or-month-first"
    if re.match(r"^[A-Za-z]{3,9}\s+\d{1,2}[,\s]+\d{2,4}", text):
        return "month-name"
    if re.match(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}", text):
        return "day-month-name"
    return "other"


def _date_column_profile(series, column_name):
    """Return a conservative date profile without modifying the dataframe."""
    if pd.api.types.is_datetime64_any_dtype(series):
        non_empty = int(series.notna().sum())
        return {"candidate": non_empty > 0, "safe": non_empty > 0, "inconsistent": False,
                "non_empty": non_empty, "parseable": non_empty, "formats": ["datetime"]}

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return {"candidate": False, "safe": False, "inconsistent": False,
                "non_empty": 0, "parseable": 0, "formats": []}

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return {"candidate": False, "safe": False, "inconsistent": False,
                "non_empty": 0, "parseable": 0, "formats": []}

    parsed = parse_dates_safely(values)
    parseable = int(parsed.notna().sum())
    total = int(len(values))
    parse_ratio = parseable / max(1, total)
    signatures = sorted({_date_signature(v) for v in values.head(250) if _date_signature(v) != "other"})
    name_hint = any(token in str(column_name).lower() for token in DATE_HINTS)
    pattern_ratio = sum(_date_signature(v) != "other" for v in values.head(250)) / max(1, min(total, 250))
    candidate = (name_hint and parse_ratio >= 0.6) or (pattern_ratio >= 0.8 and parse_ratio >= 0.8)
    return {
        "candidate": candidate,
        "safe": candidate and parse_ratio >= 0.8,
        "inconsistent": candidate and (len(signatures) > 1 or (0 < parseable < total)),
        "non_empty": total,
        "parseable": parseable,
        "formats": signatures,
    }


def build_data_quality_summary(df):
    """Return a read-only quality score for the dashboard; it never changes data."""
    rows, columns = int(df.shape[0]), int(df.shape[1])
    total_cells = max(1, rows * max(1, columns))
    null_cells = int(df.isna().sum().sum())
    blank_cells = 0
    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna().astype(str)
        blank_cells += int(values.str.strip().eq("").sum())

    missing_cells = null_cells + blank_cells
    missing_pct = (missing_cells / total_cells) * 100
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / max(1, rows)) * 100
    missing_columns = int((df.isna().sum() > 0).sum())

    deductions = []
    if missing_pct:
        deductions.append({"label": "Missing or blank values", "points": round(min(40, missing_pct * 0.8), 1)})
    if duplicate_pct:
        deductions.append({"label": "Duplicate rows", "points": round(min(25, duplicate_pct * 0.7), 1)})
    score = max(0, int(round(100 - sum(item["points"] for item in deductions))))
    label = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Needs attention"

    return {
        "score": score,
        "label": label,
        "metrics": {
            "completeness": round(100 - missing_pct, 1),
            "duplicate_rows": duplicate_rows,
            "blank_cells": blank_cells,
            "missing_columns": missing_columns,
        },
        "deductions": deductions,
    }


def build_dashboard_payload(store, df=None):
    """One response shape for dashboard uploads and data refreshes."""
    df = store["df"] if df is None else df
    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    charts = compute_charts(df, det)
    insights = auto_insights(df, det, kpis)
    quality = build_data_quality_summary(df)
    missing = {str(c): int(v) for c, v in df.isnull().sum().items() if int(v) > 0}
    date_range = {}
    if det["date_col"] and det["date_col"] in df.columns:
        dates = pd.to_datetime(df[det["date_col"]], errors="coerce").dropna()
        if not dates.empty:
            date_range = {"min": str(dates.min().date()), "max": str(dates.max().date()), "col": det["date_col"]}

    return {
        "success": True,
        "filename": store.get("filename") or "data.csv",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_rows": int(df.shape[0]),
        "filtered_rows": int(df.shape[0]),
        "col_names": list(df.columns),
        "analysis": store.get("analysis_text", ""),
        "col_stats": build_col_stats(df, det["num_cols"]),
        "charts": charts,
        "kpis": kpis,
        "insights": insights,
        "missing": missing,
        "quality": quality,
        "detected": det,
        "filter_options": build_filter_options(df, det),
        "date_range": date_range,
        "table_data": df.head(300).fillna("").to_dict(orient="records"),
        "table_cols": list(df.columns),
        "all_numeric": det["num_cols"],
        "all_cat": det["cat_cols"],
        "ai_enabled": bool(API_KEY),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    store = get_user_store()

    try:
        # Keep uploaded values intact; chart helpers handle date detection
        # without modifying the source dataframe.
        df = read_uploaded_file(file)

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        store["df"] = df.copy(deep=True)
        store["filtered"] = df.copy()
        store["filename"] = file.filename

        try:
            describe_txt = df.describe(include="all").fillna("").astype(str).head(10).to_string()
        except Exception:
            describe_txt = "Summary unavailable"

        analysis = call_ai(
            "You are a data analyst. Give a short plain-text summary in max 5 lines. No markdown.",
            f"Rows: {df.shape[0]}\nColumns: {list(df.columns)}\nSummary:\n{describe_txt}\nGive key findings and one recommendation.",
            260
        )
        store["analysis_text"] = analysis

        response_data = build_dashboard_payload(store)

        return jsonify(make_json_safe(response_data))

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Could not process file: {str(e)}"}), 500


@app.route("/filter", methods=["POST"])
def apply_filter():
    store = get_user_store()
    if store["df"] is None:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    base_df = store["df"]
    df = base_df.copy()

    for col, val in body.get("filters", {}).items():
        if col in df.columns and val not in [None, "", "__ALL__"]:
            df = df[df[col].astype(str) == str(val)]

    dr = body.get("date_range", {})
    date_col = dr.get("col")
    if date_col and date_col in df.columns:
        try:
            date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("from"):
                df = df[date_series >= pd.to_datetime(dr["from"])]
                date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("to"):
                df = df[date_series <= pd.to_datetime(dr["to"])]
        except Exception:
            pass

    store["filtered"] = df.copy()

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    charts = compute_charts(df, det)
    insights = auto_insights(df, det, kpis)
    col_stats = build_col_stats(df, det["num_cols"])
    quality = build_data_quality_summary(df)

    response_data = {
        "rows": int(len(df)),
        "total_rows": int(len(base_df)),
        "filtered_rows": int(len(df)),
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
        "quality": quality,
        "col_stats": col_stats,
        "table_cols": list(df.columns),
        "table_data": df.head(300).fillna("").to_dict(orient="records")
    }

    return jsonify(make_json_safe(response_data))


def _normalize_column_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _mentioned_columns(prompt, columns):
    normalized_prompt = _normalize_column_name(prompt)
    matches = []
    for col in columns:
        normalized = _normalize_column_name(col)
        if normalized and normalized in normalized_prompt:
            matches.append(str(col))
    return matches


def _singular_word(value):
    """Normalize simple English plural forms for column-name matching."""
    value = str(value).lower()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us")):
        return value[:-1]
    return value


def _column_words(column):
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column))
    return [_singular_word(word) for word in re.findall(r"[a-z0-9]+", spaced.lower())]


def _question_columns(question, columns):
    """Resolve explicitly mentioned columns across camelCase, spaces, and plurals."""
    question_text = str(question).lower()
    question_compact = _normalize_column_name(question_text)
    question_words = {_singular_word(word) for word in re.findall(r"[a-z0-9]+", question_text)}
    candidates = []

    for index, column in enumerate(columns):
        column_name = str(column)
        compact = _normalize_column_name(column_name)
        words = set(_column_words(column_name))
        if not compact or not words:
            continue

        compact_forms = {compact}
        if len(compact) > 3 and compact.endswith("s"):
            compact_forms.add(compact[:-1])
        else:
            compact_forms.add(compact + "s")

        if any(len(form) >= 3 and form in question_compact for form in compact_forms):
            candidates.append((1000 + len(compact), index, column_name))
        elif words.issubset(question_words):
            candidates.append((100 + len(words), index, column_name))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, _, column in candidates]


# ------------------------------------------------------------------
# ASK ANYTHING: validated, full-data query engine
# ------------------------------------------------------------------
# The chat must never calculate an answer from a preview/sample of the
# dataframe.  These helpers convert common natural-language questions into a
# small, safe query specification and execute only known Pandas operations.

_QUERY_OPERATORS = {"equals", "not_equals", "contains", "gt", "gte", "lt", "lte"}
_QUERY_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count"}
_QUERY_LIMIT_MAX = 50


def _is_query_numeric(series):
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    return _query_numeric_series(sample).notna().mean() >= 0.8


def _query_numeric_series(series):
    """Safely coerce number-like upload values such as `₹8,999.50` for Q&A only."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str).str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"^[₹$€£]\s*", "", regex=True)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _is_query_date(series, column_name):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    name_hint = any(token in str(column_name).lower() for token in DATE_HINTS)
    if not name_hint:
        sample = series.dropna().astype(str).str.strip().head(20)
        if sample.empty:
            return False
        date_like = sum(_date_signature(value) != "other" for value in sample)
        if date_like / len(sample) < 0.8:
            return False
    return _date_column_profile(series, column_name)["candidate"]


def _rank_query_columns(question, columns):
    """Rank flexible column references without accepting arbitrary column names."""
    question_text = str(question).lower()
    question_compact = _normalize_column_name(question_text)
    question_words = {_singular_word(word) for word in re.findall(r"[a-z0-9]+", question_text)}
    ranked = []

    for index, column in enumerate(columns):
        name = str(column)
        compact = _normalize_column_name(name)
        words = set(_column_words(name))
        if not compact or not words:
            continue

        if len(compact) >= 3 and compact in question_compact:
            score = 1000 + len(compact)
        elif words.issubset(question_words):
            score = 200 + len(words) * 10
        else:
            matches = words.intersection(question_words)
            # "top product" should naturally prefer ProductName over
            # ProductID, while still working for arbitrary column names.
            first_word = _column_words(name)[0] if _column_words(name) else ""
            if not matches or first_word not in question_words:
                continue
            score = len(matches) * 15
            if "name" in words and any(word in question_words for word in ["top", "best", "show", "list", "which"]):
                score += 9
            if len(words) == 1:
                score += 5
        ranked.append((score, index, name))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, _, name in ranked]


def _query_column(question, df, kind="any", exclude=None):
    """Return the best real dataframe column matching the question."""
    excluded = {str(value) for value in (exclude or [])}
    for column in _rank_query_columns(question, list(df.columns)):
        if column in excluded:
            continue
        series = df[column]
        if kind == "numeric" and not _is_query_numeric(series):
            continue
        if kind == "date" and not _is_query_date(series, column):
            continue
        if kind == "dimension" and (_is_query_numeric(series) or _is_query_date(series, column)):
            continue
        return column
    return None


def _query_columns(question, df, kind="any"):
    results = []
    for column in _rank_query_columns(question, list(df.columns)):
        series = df[column]
        if kind == "numeric" and not _is_query_numeric(series):
            continue
        if kind == "date" and not _is_query_date(series, column):
            continue
        if kind == "dimension" and (_is_query_numeric(series) or _is_query_date(series, column)):
            continue
        results.append(column)
    return results


def _first_query_date_column(df):
    for column in df.columns:
        if _is_query_date(df[column], str(column)):
            return str(column)
    return None


def _column_phrase_pattern(column):
    words = _column_words(column)
    if not words:
        return ""
    return r"(?:" + r"[\s_-]*".join(re.escape(word) for word in words) + r")"


def _numeric_filters_from_question(question, df):
    """Find safe numeric comparisons such as `Price > 5000` or `price above 5000`."""
    text = str(question).lower()
    filters = []
    word_operators = {
        "above": "gt", "over": "gt", "greater than": "gt", "more than": "gt",
        "below": "lt", "under": "lt", "less than": "lt", "fewer than": "lt",
        "at least": "gte", "minimum": "gte", "at most": "lte", "maximum": "lte",
    }
    symbol_operators = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "=": "equals", "==": "equals"}

    for column in df.columns:
        if not _is_query_numeric(df[column]):
            continue
        phrase = _column_phrase_pattern(column)
        if not phrase:
            continue
        found = None
        symbol_match = re.search(
            rf"\b{phrase}\b\s*(>=|<=|==|=|>|<)\s*[₹$]?\s*(-?[\d,]+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if symbol_match:
            found = (symbol_operators[symbol_match.group(1)], symbol_match.group(2))
        else:
            words_match = re.search(
                rf"\b{phrase}\b\s*(above|over|greater than|more than|below|under|less than|fewer than|at least|at most|minimum|maximum)\s*[₹$]?\s*(-?[\d,]+(?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if words_match:
                found = (word_operators[words_match.group(1).lower()], words_match.group(2))
        if found:
            try:
                filters.append({"column": str(column), "operator": found[0], "value": float(found[1].replace(",", ""))})
            except ValueError:
                pass
    return filters


def _value_filters_from_question(question, df):
    """Match categorical values against the real current data, ignoring case/space."""
    question_compact = _normalize_column_name(question)
    candidates = []
    for column in df.columns:
        series = df[column]
        if _is_query_numeric(series) or _is_query_date(series, str(column)):
            continue
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        counts = values.value_counts()
        # Scanning bounded-cardinality values is deliberate: it avoids turning a
        # long list of free-form names into accidental filters.
        if counts.empty or len(counts) > 500:
            continue
        for value, count in counts.items():
            normal = _normalize_column_name(value)
            if len(normal) >= 3 and normal in question_compact:
                candidates.append((len(normal), str(column), str(value), int(count)))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {"column": column, "operator": "equals", "value": value}
        for _, column, value, _ in candidates[:8]
    ]


def _dedupe_query_filters(filters):
    seen = set()
    output = []
    for item in filters:
        key = (str(item.get("column")), str(item.get("operator")), str(item.get("value")).casefold())
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _apply_query_filters(df, filters):
    """Apply validated filters with AND semantics, and OR for selected values of one column."""
    result = df.copy()
    equality_groups = {}
    remaining = []
    for item in filters or []:
        if item["operator"] == "equals" and not _is_query_numeric(result[item["column"]]):
            equality_groups.setdefault(item["column"], []).append(str(item["value"]))
        else:
            remaining.append(item)

    for column, values in equality_groups.items():
        normalized = result[column].fillna("").astype(str).str.strip().str.casefold()
        allowed = {value.strip().casefold() for value in values}
        result = result.loc[normalized.isin(allowed)].copy()

    for item in remaining:
        column, operator, value = item["column"], item["operator"], item["value"]
        series = result[column]
        if operator == "equals":
            mask = _query_numeric_series(series) == float(value)
        elif operator == "not_equals":
            if _is_query_numeric(series):
                mask = _query_numeric_series(series) != float(value)
            else:
                mask = series.fillna("").astype(str).str.strip().str.casefold() != str(value).strip().casefold()
        elif operator == "contains":
            mask = series.fillna("").astype(str).str.contains(str(value), case=False, regex=False)
        else:
            numeric = _query_numeric_series(series)
            comparisons = {
                "gt": numeric > float(value), "gte": numeric >= float(value),
                "lt": numeric < float(value), "lte": numeric <= float(value),
            }
            mask = comparisons[operator]
        result = result.loc[mask.fillna(False)].copy()
    return result


def _question_limit(question, default=10):
    match = re.search(r"\b(?:top|bottom|first|last|show|list)\s+(\d{1,3})\b", str(question).lower())
    if not match:
        return default
    return max(1, min(_QUERY_LIMIT_MAX, int(match.group(1))))


def _query_aggregation(question, metric=None, grouped=False):
    text = str(question).lower()
    metric_words = set(_column_words(metric or ""))
    # Repeated transactional measures are normally meaningful as totals when
    # ranking or grouping (for example, sales by city or product by quantity).
    if grouped and metric_words.intersection({"sale", "revenue", "amount", "value", "income", "quantity", "unit", "volume"}):
        return "sum"
    if any(word in text for word in ["average", "avg", "mean"]):
        return "mean"
    if "median" in text:
        return "median"
    if any(word in text for word in ["minimum", "lowest", "min "]):
        return "min"
    if any(word in text for word in ["maximum", "highest", "max "]):
        return "max"
    if any(word in text for word in ["total", "sum"]):
        return "sum"
    # For ranking a raw measure such as Price, average is a transparent default;
    # sales/revenue-like questions and grouped analyses conventionally use totals.
    return "mean"


def _group_column_from_question(question, df, dimension_columns):
    text = str(question)
    by_match = re.search(r"\bby\s+(.+)", text, flags=re.IGNORECASE)
    if by_match:
        tail = by_match.group(1)
        if re.search(r"\b(month|monthly)\b", tail, flags=re.IGNORECASE):
            date_column = _query_column(tail, df, "date") or _first_query_date_column(df)
            if date_column:
                return date_column, "month"
        candidate = _query_column(tail, df, "dimension") or _query_column(tail, df, "date")
        if candidate:
            return candidate, None
    if dimension_columns:
        return dimension_columns[0], None
    return None, None


def _build_query_intent(question, df):
    """Deterministically parse the common analyst questions before using AI fallback."""
    text = str(question).strip()
    lower = text.lower()
    value_filters = _value_filters_from_question(text, df)
    numeric_filters = _numeric_filters_from_question(text, df)
    filters = _dedupe_query_filters(value_filters + numeric_filters)
    dimensions = _query_columns(text, df, "dimension")
    metrics = _query_columns(text, df, "numeric")
    by_match = re.search(r"\bby\s+(.+)", text, flags=re.IGNORECASE)
    # In a ranking request, the measure after "by" is the strongest signal.
    # This keeps `Top 5 products by Price` correct even when product/name is
    # also a partial match elsewhere in the question.
    metric_after_by = _query_column(by_match.group(1), df, "numeric") if by_match else None
    metric = metric_after_by or (metrics[0] if metrics else None)
    dimension = dimensions[0] if dimensions else None
    limit = _question_limit(text)
    asks_count = bool(re.search(r"\b(how many|count|number of)\b", lower))
    asks_percent = bool(re.search(r"\b(percent|percentage|share|rate)\b", lower))
    asks_rows = bool(re.search(r"\b(show|list|display|give|find)\b", lower))
    asks_breakdown = bool(re.search(r"\b(breakdown|distribution)\b", lower))
    asks_common = bool(re.search(r"\b(most common|most frequent)\b", lower))
    asks_missing = bool(re.search(r"\b(missing|null|blank|empty|incomplete)\b", lower))
    ranking = bool(re.search(r"\b(top|bottom|best|worst|highest|lowest|most|least)\b", lower))
    ascending = bool(re.search(r"\b(bottom|worst|lowest|least)\b", lower))

    # Compare two real values in one categorical column; those values are not
    # accidental sequential filters, they are two alternatives to compare.
    by_column = {}
    for item in value_filters:
        by_column.setdefault(item["column"], []).append(item["value"])
    comparison_column = next((column for column, values in by_column.items() if len(set(values)) >= 2), None)
    if comparison_column and re.search(r"\b(compare|vs\.?|versus)\b", lower):
        comparison_values = list(dict.fromkeys(by_column[comparison_column]))[:2]
        other_filters = [item for item in filters if item["column"] != comparison_column]
        return {
            "operation": "comparison", "column": comparison_column,
            "values": comparison_values, "metric": metric,
            "aggregation": _query_aggregation(text, metric, grouped=True) if metric else "count",
            "filters": other_filters,
        }

    if asks_percent and filters:
        return {"operation": "percentage", "filters": filters}

    if asks_missing:
        return {"operation": "missing_summary", "filters": filters}

    if re.search(r"\b(unique|distinct)\b", lower):
        column = dimension or metric or (list(df.columns)[0] if len(df.columns) else None)
        if column:
            return {"operation": "unique_count", "column": column, "filters": filters}

    if asks_count:
        if re.search(r"\b(unique|distinct)\b", lower):
            column = dimension or metric or (list(df.columns)[0] if len(df.columns) else None)
            if column:
                return {"operation": "unique_count", "column": column, "filters": filters}
        return {"operation": "count", "filters": filters}

    if asks_breakdown or (asks_percent and dimension and not filters):
        if dimension:
            return {"operation": "value_counts", "column": dimension, "filters": filters, "limit": limit}

    if asks_common and dimension:
        return {"operation": "most_common", "column": dimension, "filters": filters}

    if asks_rows:
        row_limit = _question_limit(text, default=20)
        if filters:
            sort_column = metric if metric and any(item["column"] == metric for item in numeric_filters) else None
            return {"operation": "rows", "filters": filters, "limit": min(20, row_limit), "sort_column": sort_column, "sort": "desc"}
        if dimension:
            return {"operation": "value_counts", "column": dimension, "filters": [], "limit": min(50, row_limit)}

    group_column, date_granularity = _group_column_from_question(text, df, dimensions)
    record_words = bool(re.search(r"\b(order|orders|record|records|row|rows|entry|entries)\b", lower))
    if ranking:
        if "best" in lower and not metric:
            label = dimension or "this field"
            return {"operation": "clarify", "message": f"Best {label} by which metric? Please choose a numeric column such as sales, revenue, quantity, price, or ask for the most frequent value."}
        if group_column and metric:
            return {
                "operation": "top", "group_by": group_column, "metric": metric,
                "aggregation": _query_aggregation(text, metric, grouped=True),
                "sort": "asc" if ascending else "desc", "limit": limit,
                "filters": filters, "date_granularity": date_granularity,
            }
        if group_column and record_words:
            return {
                "operation": "top", "group_by": group_column, "metric": None,
                "aggregation": "count", "sort": "asc" if ascending else "desc",
                "limit": limit, "filters": filters,
            }
        if metric and not group_column:
            if re.search(r"\b(highest|maximum|max|lowest|minimum|min)\b", lower):
                aggregation = "min" if ascending else "max"
                return {"operation": "aggregate", "column": metric, "aggregation": aggregation, "filters": filters}
            return {"operation": "rows", "filters": filters, "limit": limit, "sort_column": metric, "sort": "asc" if ascending else "desc"}
        if dimension:
            return {"operation": "clarify", "message": f"Top {dimension} needs a metric. For example, ask for top {dimension} by a numeric column, or ask for the most common {dimension}."}

    aggregation_words = bool(re.search(r"\b(total|sum|average|avg|mean|median|minimum|maximum|max|min)\b", lower))
    if metric and group_column and re.search(r"\bby\b", lower):
        return {
            "operation": "grouped_aggregate", "group_by": group_column, "metric": metric,
            "aggregation": _query_aggregation(text, metric, grouped=True), "filters": filters,
            "limit": limit, "date_granularity": date_granularity,
        }
    if metric and aggregation_words:
        return {"operation": "aggregate", "column": metric, "aggregation": _query_aggregation(text, metric), "filters": filters}

    # A short direct question such as "PaymentMethod?" is treated as a list of
    # available values, not as an unreliable request to inspect preview rows.
    if dimension and re.search(r"\b(list|all|available|values?)\b", lower):
        return {"operation": "value_counts", "column": dimension, "filters": filters, "limit": min(50, limit)}
    return None


def _extract_json_object(raw):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _resolve_known_query_column(value, df):
    if not isinstance(value, str):
        return None
    target = _normalize_column_name(value)
    if not target:
        return None
    for column in df.columns:
        if _normalize_column_name(column) == target:
            return str(column)
    return None


def _validate_ai_query_intent(spec, df):
    """Validate every AI-suggested operation, field, operator and limit."""
    if not isinstance(spec, dict):
        return None
    raw_operation = str(spec.get("operation", "")).lower().strip()
    aggregate_aliases = {name: name for name in _QUERY_AGGREGATIONS if name != "count"}
    operation = "aggregate" if raw_operation in aggregate_aliases else raw_operation
    allowed = {"count", "aggregate", "value_counts", "percentage", "unique_count", "top", "bottom", "grouped_aggregate", "rows", "most_common"}
    if operation not in allowed:
        return None

    intent = {"operation": operation, "filters": []}
    raw_filters = spec.get("filters", [])
    if not isinstance(raw_filters, list) or len(raw_filters) > 6:
        return None
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, dict):
            return None
        column = _resolve_known_query_column(raw_filter.get("column"), df)
        operator = str(raw_filter.get("operator", "equals")).lower()
        value = raw_filter.get("value")
        if not column or operator not in _QUERY_OPERATORS or isinstance(value, (dict, list)):
            return None
        if isinstance(value, str) and (not value.strip() or len(value) > 200):
            return None
        if operator in {"gt", "gte", "lt", "lte"}:
            if not _is_query_numeric(df[column]):
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
        intent["filters"].append({"column": column, "operator": operator, "value": value})

    for key in ["column", "metric", "group_by", "sort_column"]:
        if key in spec and spec[key] is not None:
            column = _resolve_known_query_column(spec[key], df)
            if not column:
                return None
            intent[key] = column

    if operation == "aggregate":
        intent["column"] = intent.get("column")
        if not intent["column"] or not _is_query_numeric(df[intent["column"]]):
            return None
        intent["aggregation"] = aggregate_aliases.get(raw_operation, str(spec.get("aggregation", "mean")).lower())
        if intent["aggregation"] not in _QUERY_AGGREGATIONS:
            return None
    if operation in {"value_counts", "unique_count", "most_common"} and not intent.get("column"):
        return None
    if operation in {"top", "bottom", "grouped_aggregate"}:
        if not intent.get("group_by"):
            return None
        if intent.get("metric") and not _is_query_numeric(df[intent["metric"]]):
            return None
        aggregation = str(spec.get("aggregation", "count" if not intent.get("metric") else "mean")).lower()
        if aggregation not in _QUERY_AGGREGATIONS:
            return None
        intent["aggregation"] = aggregation
    if operation == "rows" and intent.get("sort_column") and not _is_query_numeric(df[intent["sort_column"]]):
        return None

    try:
        limit = int(spec.get("limit", 10))
    except (TypeError, ValueError):
        return None
    intent["limit"] = max(1, min(_QUERY_LIMIT_MAX, limit))
    default_direction = "asc" if raw_operation == "bottom" else "desc"
    direction = str(spec.get("sort", default_direction)).lower()
    intent["sort"] = direction if direction in {"asc", "desc"} else default_direction
    return intent


def _ai_query_intent(question, df):
    """Optional AI interpretation fallback.  It receives schema only, never data rows."""
    if not API_KEY or df.empty:
        return None
    schema = []
    for column in list(df.columns)[:60]:
        series = df[column]
        schema.append({
            "name": str(column), "type": str(series.dtype), "non_missing": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True)),
        })
    raw = call_ai(
        "Return one JSON object only. Interpret the question; never calculate values and never return code. "
        "Allowed operations: count, sum, mean, median, min, max, value_counts, percentage, unique_count, top, bottom, grouped_aggregate, rows, most_common. "
        "Use exact column names from the schema. Filters may use equals, not_equals, contains, gt, gte, lt, lte. "
        "For unclear ranking requests return null.",
        f"Dataset schema: {json.dumps(schema, ensure_ascii=False)}\nQuestion: {question}",
        260,
    )
    return _validate_ai_query_intent(_extract_json_object(raw), df)


def _format_query_number(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isclose(number, round(number)):
        return f"{int(round(number)):,}"
    return f"{number:,.2f}"


def _describe_query_filters(filters):
    if not filters:
        return "the current working dataset"
    pieces = []
    operator_labels = {"equals": "=", "not_equals": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "contains": "contains"}
    for item in filters:
        pieces.append(f"{item['column']} {operator_labels.get(item['operator'], item['operator'])} {item['value']}")
    return "; ".join(pieces)


def _query_table_preview(df, limit):
    total = len(df)
    preview = df.head(limit).copy()
    columns = list(preview.columns)[:12]
    preview = preview[columns].fillna("")
    table = preview.to_string(index=False, max_colwidth=26)
    suffix = "" if len(df.columns) <= len(columns) else f"\n(Displaying the first {len(columns)} columns.)"
    return f"Showing first {min(limit, total):,} of {total:,} matching records:\n{table}{suffix}"


def _execute_query_intent(intent, df):
    """Execute a previously validated specification on the full active dataframe."""
    operation = intent["operation"]
    if operation == "clarify":
        return intent["message"]
    filters = intent.get("filters", [])
    result = _apply_query_filters(df, filters)
    count = int(len(result))

    if operation == "count":
        return f"{count:,} record{'s' if count != 1 else ''} match {_describe_query_filters(filters)}."

    if operation == "percentage":
        total = int(len(df))
        percent = (count / total * 100) if total else 0
        return f"{_describe_query_filters(filters)} represents {percent:.1f}% of the current working dataset ({count:,} of {total:,} records)."

    if operation == "missing_summary":
        issues = []
        for column in result.columns:
            null_count = int(result[column].isna().sum())
            blank_count = 0
            if pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column]):
                text = result[column].dropna().astype(str)
                blank_count = int(text.str.strip().eq("").sum())
            total_missing = null_count + blank_count
            if total_missing:
                issues.append((str(column), total_missing, null_count, blank_count))
        if not issues:
            return f"No missing or blank values were found in {_describe_query_filters(filters)}."
        issues.sort(key=lambda item: (-item[1], item[0].lower()))
        total_missing = sum(item[1] for item in issues)
        lines = [f"{total_missing:,} missing or blank value{'s' if total_missing != 1 else ''} found across {len(issues):,} column{'s' if len(issues) != 1 else ''}:"]
        for column, total, null_count, blank_count in issues[:15]:
            detail = []
            if null_count:
                detail.append(f"{null_count:,} missing")
            if blank_count:
                detail.append(f"{blank_count:,} blank")
            lines.append(f"• {column} — {total:,} ({', '.join(detail)})")
        if len(issues) > 15:
            lines.append(f"Showing 15 of {len(issues):,} affected columns.")
        return "\n".join(lines)

    if operation == "unique_count":
        column = intent["column"]
        unique = int(result[column].dropna().astype(str).str.strip().replace("", np.nan).nunique())
        return f"{column} has {unique:,} unique non-empty value{'s' if unique != 1 else ''} in {_describe_query_filters(filters)}."

    if operation in {"value_counts", "most_common"}:
        column = intent["column"]
        values = result[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        counts = values.value_counts()
        if counts.empty:
            return f"No non-empty {column} values were found in {_describe_query_filters(filters)}."
        if operation == "most_common":
            value, value_count = counts.index[0], int(counts.iloc[0])
            percentage = value_count / int(counts.sum()) * 100
            return f"The most common {column} is {value} — {value_count:,} record{'s' if value_count != 1 else ''} ({percentage:.1f}% of non-empty {column} values)."
        limit = min(intent.get("limit", 10), len(counts))
        total = int(counts.sum())
        lines = [f"{column} breakdown ({total:,} non-empty records):"]
        lines.extend(
            f"{index}. {value} — {int(value_count):,} ({int(value_count) / total * 100:.1f}%)"
            for index, (value, value_count) in enumerate(counts.head(limit).items(), start=1)
        )
        if len(counts) > limit:
            lines.append(f"Showing {limit} of {len(counts):,} unique values.")
        return "\n".join(lines)

    if operation == "aggregate":
        column = intent["column"]
        values = _query_numeric_series(result[column]).dropna()
        if values.empty:
            return f"{column} has no numeric values in {_describe_query_filters(filters)}."
        aggregation = intent["aggregation"]
        value = getattr(values, aggregation)() if aggregation != "count" else values.count()
        labels = {"sum": "total", "mean": "average", "median": "median", "min": "minimum", "max": "maximum", "count": "count"}
        return f"The {labels[aggregation]} {column} is {_format_query_number(value)} across {len(values):,} non-missing records."

    if operation == "comparison":
        column = intent["column"]
        choices = [str(value).strip().casefold() for value in intent["values"]]
        values = result[column].fillna("").astype(str).str.strip().str.casefold()
        compared = result.loc[values.isin(choices)].copy()
        if compared.empty:
            return f"No records matched the selected {column} values."
        metric = intent.get("metric")
        aggregation = intent.get("aggregation", "count")
        lines = [f"Comparison by {column}:"]
        for requested in intent["values"]:
            subset = compared.loc[values == str(requested).strip().casefold()]
            if metric and aggregation != "count":
                numeric = _query_numeric_series(subset[metric]).dropna()
                value = getattr(numeric, aggregation)() if not numeric.empty else np.nan
                lines.append(f"{requested} — {_format_query_number(value)} {aggregation} {metric} ({len(subset):,} records)")
            else:
                lines.append(f"{requested} — {len(subset):,} records")
        return "\n".join(lines)

    if operation in {"top", "bottom", "grouped_aggregate"}:
        group_by = intent["group_by"]
        metric = intent.get("metric")
        aggregation = intent.get("aggregation", "count")
        working = result.copy()
        grouping = group_by
        if intent.get("date_granularity") == "month":
            parsed = pd.to_datetime(working[group_by], errors="coerce")
            working = working.loc[parsed.notna()].copy()
            working["__query_month__"] = parsed.loc[parsed.notna()].dt.to_period("M").astype(str)
            grouping = "__query_month__"
        if metric and aggregation != "count":
            numeric = _query_numeric_series(working[metric])
            working = working.loc[numeric.notna()].copy()
            working["__query_metric__"] = numeric.loc[numeric.notna()]
            grouped = working.groupby(grouping, dropna=True)["__query_metric__"].agg(aggregation)
        else:
            grouped = working.groupby(grouping, dropna=True).size()
        grouped = grouped.dropna().sort_values(ascending=intent.get("sort", "desc") == "asc")
        if grouped.empty:
            return f"No grouped results were found for {group_by}."
        limit = min(intent.get("limit", 10), len(grouped))
        label = str(group_by) if grouping == group_by else f"{group_by} month"
        if operation == "grouped_aggregate":
            heading = f"{aggregation.title()} {metric} by {label}:"
        else:
            direction = "Bottom" if intent.get("sort") == "asc" else "Top"
            metric_label = f"{aggregation} {metric}" if metric else "record count"
            heading = f"{direction} {limit} {label} by {metric_label}:"
        lines = [heading]
        lines.extend(f"{index}. {name} — {_format_query_number(value)}" for index, (name, value) in enumerate(grouped.head(limit).items(), start=1))
        return "\n".join(lines)

    if operation == "rows":
        sorted_result = result
        sort_column = intent.get("sort_column")
        if sort_column:
            sorted_result = result.assign(__query_sort__=_query_numeric_series(result[sort_column])).sort_values(
                "__query_sort__", ascending=intent.get("sort") == "asc", na_position="last"
            ).drop(columns="__query_sort__")
        if sorted_result.empty:
            return f"No records match {_describe_query_filters(filters)}."
        return _query_table_preview(sorted_result, intent.get("limit", 20))

    return "I could not run that request safely. Please ask using a column name or a simple filter."


def _requested_chart_type(prompt):
    text = prompt.lower()
    if any(word in text for word in ["doughnut", "donut"]):
        return "doughnut"
    if "pie" in text:
        return "pie"
    if any(word in text for word in ["scatter", "correlation"]):
        return "scatter"
    if any(word in text for word in ["line", "trend", "over time", "monthly"]):
        return "line"
    if any(word in text for word in ["bar", "column"]):
        return "bar"
    return None


def _chart_aggregation(prompt):
    text = prompt.lower()
    if any(word in text for word in ["average", "avg", "mean"]):
        return "mean"
    if any(word in text for word in ["count", "distribution", "how many", "number of"]):
        return "count"
    return "sum"


def _chart_limit(prompt):
    match = re.search(r"\btop\s+(\d{1,2})\b", prompt.lower())
    return min(30, max(1, int(match.group(1)))) if match else 10


def _validate_chart_spec(spec, df):
    if not isinstance(spec, dict):
        return None
    chart_type = str(spec.get("chart_type", "")).lower()
    aggregation = str(spec.get("aggregation", "")).lower()
    x_col = spec.get("x_col")
    y_col = spec.get("y_col")
    if chart_type not in {"bar", "line", "pie", "doughnut", "scatter"} or x_col not in df.columns:
        return None
    if aggregation not in {"sum", "mean", "count"}:
        return None
    if chart_type == "scatter":
        if y_col not in df.columns or not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            return None
    elif aggregation != "count":
        if y_col not in df.columns or not pd.api.types.is_numeric_dtype(df[y_col]):
            return None
    limit = spec.get("limit", 10)
    try:
        limit = min(30, max(1, int(limit)))
    except (ValueError, TypeError):
        limit = 10
    return {
        "chart_type": chart_type,
        "x_col": str(x_col),
        "y_col": str(y_col) if y_col in df.columns else None,
        "aggregation": aggregation,
        "sort": "asc" if str(spec.get("sort", "desc")).lower() == "asc" else "desc",
        "limit": limit,
        "title": str(spec.get("title") or "Generated chart")[:120],
    }


def _ai_chart_spec(prompt, df):
    if not API_KEY:
        return None
    column_context = []
    for column in list(df.columns)[:30]:
        series = df[column]
        item = {
            "name": str(column),
            "dtype": str(series.dtype),
            "non_empty": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.dropna()
            if not numeric.empty:
                item["range"] = [float(numeric.min()), float(numeric.max())]
        column_context.append(item)
    raw = call_ai(
        "Return ONLY one JSON object. Never return code. Keys: chart_type, x_col, y_col, aggregation, sort, limit, title. "
        "chart_type must be bar, line, pie, doughnut, or scatter. aggregation must be sum, mean, or count.",
        f"Request: {prompt}\nAvailable dataset columns (use names exactly): {json.dumps(column_context)}",
        180,
        timeout_seconds=8,
        max_attempts=1,
    )
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return _validate_chart_spec(json.loads(match.group(0)), df)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _is_comparison_request(prompt):
    text = str(prompt).lower()
    return bool(re.search(r"\b(vs|versus|against|compare|comparison)\b", text))


def _category_value_comparison(prompt, df, category_cols):
    """Find requests such as 'Electronics vs Clothing' in any categorical column."""
    if not _is_comparison_request(prompt):
        return None

    prompt_compact = _normalize_column_name(prompt)
    matches = []
    for column in category_cols:
        values = df[column].dropna().astype(str).str.strip().unique().tolist()
        selected = []
        for value in values:
            normalized = _normalize_column_name(value)
            forms = {normalized, _singular_word(normalized)}
            positions = [prompt_compact.find(form) for form in forms if len(form) >= 3 and form in prompt_compact]
            if positions:
                selected.append((min(positions), str(value)))
        if len(selected) >= 2:
            selected.sort(key=lambda item: item[0])
            matches.append((len(selected), str(column), [value for _, value in selected]))

    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1].lower()))
    _, column, selected = matches[0]
    return column, selected


def _infer_chart_spec(prompt, df):
    columns = [str(c) for c in df.columns]
    mentioned = _question_columns(prompt, columns) or _mentioned_columns(prompt, columns)
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    # Avoid parsing every value of every text column on large uploads.  The
    # lightweight check only fully parses columns that look date-like first.
    date_cols = [c for c in columns if _is_query_date(df[c], c)]
    category_cols = [
        c for c in columns
        if c not in numeric_cols and c not in date_cols and 1 < int(df[c].nunique(dropna=True)) <= 60
    ]
    category_cols.sort(key=lambda column: (int(df[column].nunique(dropna=True)), column.lower()))
    requested_type = _requested_chart_type(prompt)
    aggregation = _chart_aggregation(prompt)
    limit = _chart_limit(prompt)
    sort = "asc" if any(word in prompt.lower() for word in ["bottom", "lowest", "ascending"]) else "desc"
    mentioned_numeric = [c for c in mentioned if c in numeric_cols]
    mentioned_date = [c for c in mentioned if c in date_cols]
    mentioned_category = [c for c in mentioned if c in category_cols]
    category_comparison = _category_value_comparison(prompt, df, category_cols)
    comparison_requested = _is_comparison_request(prompt)

    if category_comparison:
        category, selected_values = category_comparison
        comparison_aggregation = aggregation if mentioned_numeric else "count"
        comparison_measure = mentioned_numeric[0] if mentioned_numeric else None
        label = " vs ".join(selected_values)
        title_prefix = "Count" if comparison_aggregation == "count" else f"{comparison_aggregation.title()} {comparison_measure}"
        return {
            "chart_type": "bar", "x_col": category, "y_col": comparison_measure,
            "aggregation": comparison_aggregation, "sort": "desc", "limit": len(selected_values),
            "include_values": selected_values, "title": f"{title_prefix}: {label}",
        }, None

    numeric_comparison = comparison_requested and (
        len(mentioned_numeric) >= 2 or "numeric" in prompt.lower() or "number" in prompt.lower()
    )
    if requested_type == "scatter" or numeric_comparison:
        pair = mentioned_numeric[:2] if len(mentioned_numeric) >= 2 else numeric_cols[:2]
        if len(pair) < 2:
            return None, "I couldn't find two numeric columns for a scatter chart. Try asking for a category distribution instead."
        return {"chart_type": "scatter", "x_col": pair[0], "y_col": pair[1], "aggregation": "count",
                "sort": sort, "limit": min(limit, 300), "title": f"{pair[1]} vs {pair[0]}"}, None

    # Never turn an unrecognised comparison into a chart for unrelated default
    # columns. For example, "Electronics vs Clothing" must either resolve to
    # those two Category values above or ask the user for a clearer value.
    if comparison_requested:
        return None, (
            "I couldn't match those two values in the current dataset. "
            "Use the category names exactly as they appear in the table, or ask to compare two numeric columns."
        )

    measure = mentioned_numeric[0] if mentioned_numeric else (numeric_cols[0] if numeric_cols else None)
    time_requested = any(word in prompt.lower() for word in ["month", "monthly", "date", "time", "trend", "over time"])
    date_col = mentioned_date[0] if mentioned_date else (date_cols[0] if date_cols and time_requested else None)
    category = mentioned_category[0] if mentioned_category else (category_cols[0] if category_cols else None)

    if date_col and (time_requested or requested_type == "line"):
        chart_type = requested_type or "line"
        aggregation = "count" if not measure else aggregation
        return {"chart_type": chart_type, "x_col": date_col, "y_col": measure, "aggregation": aggregation,
                "sort": "asc", "limit": min(limit, 24),
                "title": f"Monthly {'count' if aggregation == 'count' else measure} by {date_col}"}, None

    if category:
        if not measure:
            aggregation = "count"
        chart_type = requested_type or ("doughnut" if aggregation == "count" and int(df[category].nunique(dropna=True)) <= 8 else "bar")
        if chart_type == "scatter":
            return None, "A scatter chart needs two numeric columns. Try a bar or doughnut chart for this category."
        return {"chart_type": chart_type, "x_col": category, "y_col": measure, "aggregation": aggregation,
                "sort": sort, "limit": limit,
                "title": f"{'Count' if aggregation == 'count' else aggregation.title() + ' ' + measure} by {category}"}, None

    if len(numeric_cols) >= 2:
        return {"chart_type": "scatter", "x_col": numeric_cols[0], "y_col": numeric_cols[1], "aggregation": "count",
                "sort": sort, "limit": min(limit, 300), "title": f"{numeric_cols[1]} vs {numeric_cols[0]}"}, None
    return None, "I couldn't find enough compatible columns for that chart. Try asking for a category distribution or compare two numeric columns."


def build_generated_chart(df, spec):
    if spec["chart_type"] == "scatter":
        points = df[[spec["x_col"], spec["y_col"]]].dropna().head(spec["limit"])
        return {
            "type": "scatter", "x": [float(v) for v in points[spec["x_col"]]],
            "y": [float(v) for v in points[spec["y_col"]]], "title": spec["title"],
            "interpretation": f"Each point compares {spec['y_col']} with {spec['x_col']}.", "spec": spec,
        }

    work = df.copy()
    x_col = spec["x_col"]
    is_date_column = _is_query_date(work[x_col], x_col)
    if is_date_column:
        dates = parse_dates_safely(work[x_col])
        work = work.loc[dates.notna()].copy()
        work["_chart_group"] = dates.loc[dates.notna()].dt.to_period("M").astype(str)
    else:
        work = work[work[x_col].notna()].copy()
        work["_chart_group"] = work[x_col].astype(str)

    included_values = [str(value) for value in spec.get("include_values", [])]
    if included_values and not is_date_column:
        allowed = {value.casefold() for value in included_values}
        work = work[work["_chart_group"].str.strip().str.casefold().isin(allowed)].copy()

    if work.empty:
        raise ValueError("There are no usable values for the selected chart columns.")
    if spec["aggregation"] == "count":
        grouped = work.groupby("_chart_group").size()
    else:
        grouped = work.groupby("_chart_group")[spec["y_col"]].agg(spec["aggregation"])
    grouped = grouped.dropna()
    if included_values and not is_date_column:
        requested_order = {value.casefold(): index for index, value in enumerate(included_values)}
        grouped = grouped.sort_index(key=lambda labels: labels.str.casefold().map(requested_order))
    else:
        grouped = grouped.sort_index() if is_date_column else grouped.sort_values(ascending=spec["sort"] == "asc")
    grouped = grouped.head(spec["limit"])
    if grouped.empty:
        raise ValueError("The requested chart does not have enough data after aggregation.")
    return {
        "type": spec["chart_type"], "labels": [str(v) for v in grouped.index],
        "values": [float(v) for v in grouped.values], "title": spec["title"],
        "interpretation": f"{spec['aggregation'].title()} values grouped by {x_col}.", "spec": spec,
    }


@app.route("/generate-chart", methods=["POST"])
def generate_chart():
    store = get_user_store()
    if store["df"] is None:
        return jsonify({"error": "Upload file first"}), 400
    prompt = str((request.json or {}).get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Describe the chart you want to create"}), 400
    if len(prompt) > 500:
        return jsonify({"error": "Please keep the chart request under 500 characters"}), 400

    df = store["filtered"] if store["filtered"] is not None else store["df"]
    inferred_spec, inference_error = _infer_chart_spec(prompt, df)
    # The deterministic spec uses the full active dataframe and covers common
    # requests immediately.  Do not make users wait for an external AI service
    # when a safe local answer is already available.
    if inferred_spec:
        spec, source = inferred_spec, "Smart analysis"
    else:
        spec = _ai_chart_spec(prompt, df)
        source = "AI" if spec else "Smart analysis"
        if not spec:
            spec = inferred_spec
            if inference_error:
                return jsonify({"error": inference_error}), 400
    try:
        chart = build_generated_chart(df, spec)
        return jsonify(make_json_safe({"chart": chart, "source": source}))
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/chart-data", methods=["POST"])
def chart_data_api():
    store = get_user_store()
    if store["df"] is None:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    df = store["filtered"] if store["filtered"] is not None else store["df"]

    cat_col = body.get("cat")
    num_col = body.get("num")
    agg = body.get("agg", "mean")

    if not cat_col or cat_col not in df.columns:
        return jsonify({"error": "Invalid category column"}), 400

    if agg != "count" and (not num_col or num_col not in df.columns):
        return jsonify({"error": "Invalid numeric column"}), 400

    try:
        fn = {"mean": "mean", "sum": "sum", "count": "count"}.get(agg, "mean")

        if agg == "count":
            grp = df.groupby(cat_col).size().sort_values(ascending=False).head(10)
        else:
            grp = getattr(df.groupby(cat_col)[num_col], fn)().round(0).sort_values(ascending=False).head(10)

        return jsonify(make_json_safe({
            "labels": list(grp.index.astype(str)),
            "values": list(grp.values)
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    store = get_user_store()
    if store["df"] is None:
        return jsonify({"error": "Upload file first"}), 400

    # Always work from the already cleaned/filtered dataframe for this session.
    # Asking a question never changes the dashboard filter state itself.
    df = store["filtered"] if store["filtered"] is not None else store["df"]
    if df.empty:
        return jsonify({"error": "The current working dataset has no rows to analyze."}), 400

    intent = _build_query_intent(question, df)
    source = "dataset"
    if intent is None:
        # AI is an optional interpreter only.  It receives the schema, not a
        # sample of records, and all returned fields are validated before Pandas
        # executes the query.
        intent = _ai_query_intent(question, df)
        source = "ai_interpreted" if intent else "dataset"

    if intent is None:
        numeric = [str(column) for column in df.columns if _is_query_numeric(df[column])][:6]
        categorical = [str(column) for column in df.columns if not _is_query_numeric(df[column])][:6]
        available = ", ".join(numeric + categorical)
        answer = (
            "I couldn't identify a safe calculation from that question. "
            f"Try naming a column or filter. Available columns include: {available or 'none'}"
        )
    else:
        try:
            answer = _execute_query_intent(intent, df)
        except (KeyError, TypeError, ValueError, pd.errors.PandasError) as error:
            answer = f"I couldn't complete that calculation safely: {error}. Please try a column name from the uploaded file."

    return jsonify(make_json_safe({
        "answer": answer,
        "has_filter": False,
        "filter_count": int(len(df)),
        "ai_enabled": bool(API_KEY),
        "source": source,
    }))


@app.route("/download-filtered")
def download_filtered():
    store = get_user_store()
    df = store["filtered"] if store["filtered"] is not None else store["df"]
    if df is None:
        return "No data uploaded yet", 400

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)

    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="filtered_data.csv"
    )


def _clean_pdf_text(text):
    """FPDF ka built-in font Unicode nahi support karta (jaise ₹, em-dash •),
    isliye PDF me daalne se pehle un characters ko safe ASCII me replace karte hain."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2022": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u20b9": "Rs.", "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "ignore").decode("latin-1")


@app.route("/download-report")
def download_report():
    store = get_user_store()
    df = store["filtered"] if store["filtered"] is not None else store["df"]
    if df is None:
        return "No data uploaded yet", 400

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    ins = auto_insights(df, det, kpis)
    fmt = request.args.get("format", "pdf").lower()

    if fmt == "txt":
        try:
            stats_txt = df.describe(include="all").fillna("").astype(str).to_string()
        except Exception:
            stats_txt = "Statistics unavailable"

        report = f"""DATALENS — AI ANALYTICS REPORT
{'=' * 60}
File: {store['filename'] or 'data.csv'}
Rows: {df.shape[0]} | Cols: {df.shape[1]}
Columns: {', '.join(df.columns)}
{'=' * 60}
AI ANALYSIS
{store['analysis_text']}
{'=' * 60}
KEY INSIGHTS
{chr(10).join('• ' + i for i in ins)}
{'=' * 60}
STATISTICS
{stats_txt}
{'=' * 60}
SAMPLE (first 20 rows)
{df.head(20).to_string(index=False)}
{'=' * 60}
Generated by DataLens AI"""

        return send_file(
            io.BytesIO(report.encode()),
            mimetype="text/plain",
            as_attachment=True,
            download_name="datalens_report.txt"
        )

    # ---- PDF report ----
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 12, "DataLens - AI Analytics Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _clean_pdf_text(f"File: {store['filename'] or 'data.csv'}   |   Rows: {df.shape[0]}   |   Columns: {df.shape[1]}"), ln=True)
    pdf.ln(4)

    def section_title(txt):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 9, _clean_pdf_text(txt), ln=True)
        pdf.set_draw_color(230, 230, 230)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(3)

    section_title("AI Analysis")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, _clean_pdf_text(store["analysis_text"] or "No AI analysis available."))
    pdf.set_x(pdf.l_margin)
    pdf.ln(4)

    section_title("Key Insights")
    pdf.set_font("Helvetica", "", 10)
    for i in ins:
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _clean_pdf_text(f"- {i}"))
    pdf.set_x(pdf.l_margin)
    pdf.ln(4)

    section_title("Column Statistics")
    col_stats = build_col_stats(df, det["num_cols"])
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 240, 250)
    headers = ["Column", "Mean", "Min", "Max", "Missing"]
    widths = [55, 35, 35, 35, 30]
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for col, s in col_stats.items():
        row = [
            _clean_pdf_text(col)[:28],
            str(s["mean"]) if s["mean"] is not None else "-",
            str(s["min"]) if s["min"] is not None else "-",
            str(s["max"]) if s["max"] is not None else "-",
            str(s["missing"]),
        ]
        for val, w in zip(row, widths):
            pdf.cell(w, 6, val, border=1)
        pdf.ln()
    pdf.ln(4)

    section_title("Sample Data (first 10 rows)")
    pdf.set_font("Helvetica", "", 8)
    sample_cols = list(df.columns)[:6]  # PDF width ke liye max 6 columns
    col_w = 190 / max(1, len(sample_cols))
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(240, 240, 250)
    for c in sample_cols:
        pdf.cell(col_w, 6, _clean_pdf_text(str(c))[:20], border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.head(10).iterrows():
        for c in sample_cols:
            val = _clean_pdf_text(str(row[c]))[:20]
            pdf.cell(col_w, 6, val, border=1)
        pdf.ln()

    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Generated by DataLens AI", align="C")

    pdf_bytes = bytes(pdf.output())

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="datalens_report.pdf"
    )


@app.route("/sample")
def sample():
    csv = """deal_id,deal_date,branch,sales_employee_id,deal_status,deal_value,discount_value,booking_amount
D0000001,2025-01-15,Mumbai,E0043,Won,1500000,75000,150000
D0000002,2025-01-22,Delhi,E0042,Won,2200000,154000,220000
D0000003,2025-02-10,Pune,E0051,Lost,900000,63000,90000
D0000004,2025-02-18,Mumbai,E0060,Won,1850000,129500,185000
D0000005,2025-03-05,Hyderabad,E0011,Won,1200000,84000,120000
D0000006,2025-03-12,Delhi,E0054,Won,750000,52500,75000
D0000007,2025-03-25,Bangalore,E0053,Lost,980000,68600,98000
D0000008,2025-04-08,Mumbai,E0043,Won,1650000,115500,165000
D0000009,2025-04-15,Pune,E0042,Won,1100000,77000,110000
D0000010,2025-05-02,Hyderabad,E0060,Lost,850000,59500,85000
D0000011,2025-05-18,Delhi,E0011,Won,1950000,136500,195000
D0000012,2025-06-01,Mumbai,E0051,Won,2100000,147000,210000
D0000013,2025-06-14,Bangalore,E0054,Won,1300000,91000,130000
D0000014,2025-07-03,Pune,E0053,Lost,760000,53200,76000
D0000015,2025-07-20,Delhi,E0043,Won,1750000,122500,175000
D0000016,2025-08-05,Hyderabad,E0042,Won,1400000,98000,140000
D0000017,2025-08-19,Mumbai,E0060,Won,1900000,133000,190000
D0000018,2025-09-03,Bangalore,E0011,Lost,670000,46900,67000
D0000019,2025-09-22,Delhi,E0051,Won,2050000,143500,205000
D0000020,2025-10-10,Pune,E0054,Won,1150000,80500,115000"""

    return send_file(
        io.BytesIO(csv.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_sales.csv"
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
