from __future__ import annotations
import io, socket, threading, webbrowser
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template_string, flash, send_file

# ML
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

app = Flask(__name__)
app.secret_key = "change-me"

# ----- State -----
DF: pd.DataFrame | None = None
MODEL: CatBoostClassifier | None = None
SCORED: pd.DataFrame | None = None
INFO: dict = {}

FEATURES = [
    "utilisation", "dpd_days", "cash_credit_ratio", "cash_debit_ratio",
    "inbound_cheque_bounce_count", "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count", "outbound_cheque_bounce_amt",
    "total_amt_credit", "total_amt_debit", "no_of_banks"
]
assert "risky" not in FEATURES, "'risky' must not be in FEATURES"  # safety guard

# ---------- Utilities ----------
def _free_port(preferred=5006):
    """Return an available TCP port (tries preferred first)."""
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

def generate_dummy(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cust = np.arange(1, n + 1)
    util = np.clip(rng.normal(5, 3, n).round(2), 0, 20)
    dpd = np.clip(rng.integers(0, 400, n), 0, 365)
    cash_c = np.clip(rng.normal(0.12, 0.06, n), 0, 0.35)
    cash_d = np.clip(rng.normal(0.12, 0.06, n), 0, 0.35)
    inb_cnt = rng.integers(0, 35, n)
    inb_amt = rng.integers(0, 500_000, n)
    out_cnt = rng.integers(0, 35, n)
    out_amt = rng.integers(0, 500_000, n)
    tot_cr = rng.integers(50_000, 3_000_000, n)
    tot_db = rng.integers(50_000, 3_000_000, n)
    banks = rng.integers(1, 16, n)
    prob = (
        0.25*(util/20) + 0.15*(dpd/365) + 0.15*cash_c + 0.15*cash_d
        + 0.10*(inb_cnt/30) + 0.10*(out_cnt/30)
        + 0.10*(np.maximum(inb_amt, out_amt)/5e5)
    )
    prob = np.clip(prob, 0, 1)
    thresh = np.quantile(prob, 0.70)  # ~30% positives
    y = (prob >= thresh).astype(np.int8)
    if np.unique(y).size < 2:
        order = np.argsort(prob); y[:] = 0; y[order[-30:]] = 1
    df = pd.DataFrame({
        "customer_no": cust,
        "utilisation": util,
        "dpd_days": dpd,
        "cash_credit_ratio": cash_c,
        "cash_debit_ratio": cash_d,
        "inbound_cheque_bounce_count": inb_cnt,
        "inbound_cheque_bounce_amt": inb_amt,
        "outbound_cheque_bounce_count": out_cnt,
        "outbound_cheque_bounce_amt": out_amt,
        "total_amt_credit": tot_cr,
        "total_amt_debit": tot_db,
        "no_of_banks": banks,
        "risky": y
    })
    return df.astype("float32", errors="ignore")

def leakage_report(df: pd.DataFrame):
    """Correlation of features (excludes id/target cols) with the target 'risky'."""
    feat_cols = [c for c in df.columns if c not in ("customer_no", "risky")]
    corr = (
        df[feat_cols]
        .corr(numeric_only=True)
        .corrwith(df["risky"])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    red_flags = [c for c, v in corr.items() if abs(v) >= 0.90]
    txt = "No red-flags (|corr with target| ≥ 0.90)." if not red_flags \
          else "Potential leakage in: " + ", ".join(red_flags)
    return txt, corr.to_frame("corr_with_target")

def _has_two_classes(y: pd.Series) -> bool:
    return pd.Series(y).nunique() >= 2

def _proxy_target(df: pd.DataFrame) -> pd.Series:
    proxy = (
        0.25*(df["utilisation"]/20) + 0.15*(df["dpd_days"]/365)
        + 0.15*df["cash_credit_ratio"] + 0.15*df["cash_debit_ratio"]
        + 0.10*(df["inbound_cheque_bounce_count"]/30)
        + 0.10*(df["outbound_cheque_bounce_count"]/30)
    )
    thresh = np.nanquantile(proxy, 0.70)
    return (proxy >= thresh).astype("int8")

def performance_at_topk(y_true: np.ndarray, proba: np.ndarray, percents: list[int]) -> pd.DataFrame:
    n = len(y_true); base_rate = y_true.mean() if n > 0 else 0.0
    order = np.argsort(-proba); y_sorted = y_true[order]
    rows = []
    for k in percents:
        top = max(1, int(np.ceil(n * (k / 100.0))))
        y_top = y_sorted[:top]; tp = y_top.sum()
        precision = tp / top
        recall = tp / y_true.sum() if y_true.sum() > 0 else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0
        rows.append({
            "top_%": k, "n_cases": int(top), "positives_in_top": int(tp),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "capture_rate": round(float(recall), 4),
            "lift": round(float(lift), 2)
        })
    return pd.DataFrame(rows)

# ---------- CatBoost training + tuning ----------
def train_and_tune(df: pd.DataFrame):
    X = df[FEATURES].astype("float32")
    y = df["risky"].astype("int8")

    if not _has_two_classes(y):
        y = _proxy_target(df)
        flash("Uploaded data had one class. Created a balanced target via proxy score (70th percentile).", "warning")
    if not _has_two_classes(y):
        idx = np.argsort(X["utilisation"].to_numpy())
        y[:] = 0; y.iloc[idx[-10:]] = 1
        flash("Target still single-class; forced minimal positives for demo.", "warning")

    # Hold-out test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Validation from train for early stopping & model selection
    Xtr_i, Xva, ytr_i, yva = train_test_split(Xtr, ytr, test_size=0.20, random_state=42, stratify=ytr)

    train_pool = Pool(Xtr_i, ytr_i)
    val_pool   = Pool(Xva,   yva)

    # Lightweight grid (fast)
    param_grid = [
        {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 3.0},
        {"depth": 6, "learning_rate": 0.12, "l2_leaf_reg": 3.0},
        {"depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 5.0},
        {"depth": 8, "learning_rate": 0.12, "l2_leaf_reg": 5.0},
    ]
    base = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    best_model, best_auc, best_params = None, -1.0, None
    for p in param_grid:
        model = CatBoostClassifier(**base, **p)
        model.fit(train_pool, eval_set=val_pool)
        score = model.get_best_score().get("validation", {}).get("AUC", float("-inf"))
        if score > best_auc:
            best_auc, best_model, best_params = score, model, p

    # Finalize on combined train+val with best params (still with early stopping)
    final_model = CatBoostClassifier(**base, **best_params)
    final_model.fit(Pool(pd.concat([Xtr_i, Xva]), pd.concat([ytr_i, yva])),
                    eval_set=Pool(Xte, yte))

    # --- Metrics on test (kept for reference; not shown in UI) ---
    yhat_test = final_model.predict(Xte).astype(int)
    proba_test = final_model.predict_proba(Xte)[:, 1]
    acc = float((yhat_test == yte).mean())
    cm = confusion_matrix(yte, yhat_test)

    # Train proba for top-k (to compare)
    proba_train = final_model.predict_proba(Xtr)[:, 1]

    # performance at top k%
    percent_list = [5, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    topk_test_df = performance_at_topk(yte.values, proba_test, percent_list)
    topk_train_df = performance_at_topk(ytr.values, proba_train, percent_list)

    return final_model, {
        "best_params": best_params,
        "val_auc": float(best_auc),
        "test_acc": acc,
        "cm": cm.tolist(),  # not rendered
        # LEFT-ALIGNED tables via justify="left"
        "topk_test_html": topk_test_df.to_html(classes="table table-sm table-striped table-hover align-middle",
                                               index=False, border=0, justify="left"),
        "topk_train_html": topk_train_df.to_html(classes="table table-sm table-striped table-hover align-middle",
                                                 index=False, border=0, justify="left"),
        "ytr_n": int(len(ytr)), "yte_n": int(len(yte)),
        "train_base_rate": float(ytr.mean()), "test_base_rate": float(yte.mean()),
        "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte
    }

def importance_tables(model: CatBoostClassifier, df: pd.DataFrame, Xte: pd.DataFrame, yte: pd.Series):
    # Global importance from CatBoost
    cb_imp = model.get_feature_importance(Pool(Xte, yte), type="PredictionValuesChange")
    imp = pd.DataFrame({"feature": FEATURES, "cb_importance": cb_imp}) \
            .sort_values("cb_importance", ascending=False)

    # Permutation importance (accuracy-based; light repeats)
    pim = permutation_importance(model, Xte, yte, n_repeats=3, random_state=0)
    perm = pd.DataFrame({"feature": FEATURES, "perm_importance": pim.importances_mean}) \
              .sort_values("perm_importance", ascending=False)
    return imp, perm

def narrative_from_importance(imp_df: pd.DataFrame) -> str:
    top = imp_df.head(5).reset_index(drop=True)
    bullets = [f"{i+1}. {row.feature} (impact {row.cb_importance:.3f})" for i, row in top.iterrows()]
    return (
        "<p><b>Executive summary:</b> Model risk is driven primarily by:</p>"
        "<ul><li>" + "</li><li>".join(bullets) + "</li></ul>"
        "<p>Higher values on the leading drivers typically raise risk probability; "
        "focus mitigation and outreach on these variables.</p>"
    )

def bucket(p: float) -> str:
    if p >= 0.90: return "Very High"
    if p >= 0.80: return "High"
    if p >= 0.70: return "Moderate"
    if p >= 0.60: return "Low"
    return "No Risk"

def score_all(model: CatBoostClassifier, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype("float32")
    proba = model.predict_proba(X)[:, 1]
    out = df[["customer_no"]].copy()
    out["prob"] = np.round(proba, 4)
    out["bucket"] = [bucket(p) for p in proba]

    # SHAP-based local reasons (top-3 absolute contributors)
    shap_vals = model.get_feature_importance(Pool(X), type="ShapValues")
    shap_contrib = shap_vals[:, :-1]  # last column is expected value
    top_idx = np.argsort(-np.abs(shap_contrib), axis=1)[:, :3]
    out["reasons"] = [", ".join(FEATURES[j] for j in top_idx[i]) for i in range(len(out))]

    return out.sort_values("prob", ascending=False)

# ---------- Routes ----------
@app.route("/ping")
def ping():
    return "ok"

@app.route("/", methods=["GET"])
def home():
    rows, cols = (DF.shape if isinstance(DF, pd.DataFrame) else (0, 0))
    # LEFT-ALIGNED: sample + scored tables
    sample_html = DF.head(10).to_html(classes="table table-sm table-striped table-hover align-middle",
                                      index=False, border=0, justify="left") \
                  if isinstance(DF, pd.DataFrame) else ""
    perf = INFO if INFO else None
    scored_html = SCORED.head(20).to_html(classes="table table-sm table-striped table-hover align-middle",
                                          index=False, border=0, justify="left") \
                  if isinstance(SCORED, pd.DataFrame) else ""
    counts = (SCORED["bucket"].value_counts().to_dict()
              if isinstance(SCORED, pd.DataFrame) else {})

    return render_template_string(TEMPLATE,
                                  rows=rows, cols=cols,
                                  sample_html=sample_html,
                                  perf=perf,
                                  scored_html=scored_html,
                                  counts=counts,
                                  imp_html=INFO.get("imp_html","") if INFO else "",
                                  perm_html=INFO.get("perm_html","") if INFO else "",
                                  narrative_html=INFO.get("narrative_html","") if INFO else "",
                                  leakage_text=INFO.get("leakage_text","") if INFO else "",
                                  leakage_table=INFO.get("leakage_table","") if INFO else "")

@app.route("/generate", methods=["POST"])
def do_generate():
    global DF, MODEL, SCORED, INFO
    DF = generate_dummy(600)
    MODEL = None; SCORED = None; INFO = {}
    flash("Dummy dataset generated (balanced classes).", "info")
    return redirect(url_for("home"))

@app.route("/upload", methods=["POST"])
def do_upload():
    global DF, MODEL, SCORED, INFO
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.", "warning")
        return redirect(url_for("home"))
    DF = pd.read_csv(io.StringIO(f.stream.read().decode("utf-8")))
    MODEL = None; SCORED = None; INFO = {}
    flash("CSV uploaded.", "info")
    return redirect(url_for("home"))

@app.route("/train", methods=["POST"])
def do_train():
    global DF, MODEL, SCORED, INFO
    if DF is None:
        flash("Load dummy data or upload CSV first.", "warning")
        return redirect(url_for("home"))

    leak_txt, leak_corr_df = leakage_report(DF)
    MODEL, metrics = train_and_tune(DF)

    # Importances (use test set from metrics)
    imp_df, perm_df = importance_tables(MODEL, DF, metrics["Xte"], metrics["yte"])
    INFO = dict(metrics)
    INFO.pop("Xtr", None); INFO.pop("ytr", None); INFO.pop("Xte", None); INFO.pop("yte", None)

    # LEFT-ALIGNED: importance + leakage tables
    INFO["imp_html"] = imp_df.to_html(classes="table table-sm table-striped table-hover align-middle",
                                      index=False, border=0, justify="left")
    INFO["perm_html"] = perm_df.to_html(classes="table table-sm table-striped table-hover align-middle",
                                        index=False, border=0, justify="left")
    INFO["narrative_html"] = narrative_from_importance(imp_df)
    INFO["leakage_text"] = leak_txt
    INFO["leakage_table"] = leak_corr_df.to_html(classes="table table-sm table-striped table-hover align-middle",
                                                 border=0, justify="left")

    flash("Model trained.", "success")
    return redirect(url_for("home"))

@app.route("/score", methods=["POST"])
def do_score():
    global DF, MODEL, SCORED
    if DF is None or MODEL is None:
        flash("Train the model first.", "warning")
        return redirect(url_for("home"))
    SCORED = score_all(MODEL, DF)
    flash("Scoring complete.", "success")
    return redirect(url_for("home"))

@app.route("/download_scored", methods=["GET"])
def download_scored():
    global SCORED
    if SCORED is None:
        flash("Nothing to download. Score first.", "warning")
        return redirect(url_for("home"))
    buf = io.StringIO(); SCORED.to_csv(buf, index=False); buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv", as_attachment=True,
                     download_name="scored_portfolio.csv")

# ---------- HTML ----------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Risk Insights Studio — Flask (CatBoost)</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root{--ink:#0f172a;--soft:#e9eef6;--brand:#2563eb;--muted:#64748b}
    body{background:linear-gradient(180deg,#f6f9ff 0%,#f8fafc 60%);color:var(--ink)}
    .hero{background:linear-gradient(90deg,#2563eb,#7c3aed);border-radius:22px;color:#fff;padding:18px 20px}
    .hero small{opacity:.9}
    .card-soft{border:1px solid #edf1f7;box-shadow:0 8px 22px rgba(16,24,40,.06);background:#fff;border-radius:18px}
    /* Fallback: force left alignment for all table cells/headers */
    .table th, .table td { text-align: left !important; }
  </style>
</head>
<body class="py-4">
<div class="container">

  <div class="hero mb-4 d-flex align-items-center justify-content-between flex-wrap">
    <div>
      <h3 class="mb-1"><i class="bi bi-graph-up-arrow me-1"></i>Risk Insights Studio</h3>
      <small>End-to-end: data → train/tune → explain → score & export</small>
    </div>
    <!-- chips removed -->
  </div>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
      {% for cat,msg in messages %}
        <div class="alert alert-{{ 'info' if cat=='message' else cat }} alert-dismissible fade show" role="alert">
          {{ msg }} <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  <div class="row g-4">
    <div class="col-lg-4">
      <div class="card card-soft">
        <div class="card-body">
          <h5 class="mb-3"><i class="bi bi-hammer"></i> Actions</h5>
          <form method="post" action="{{ url_for('do_generate') }}" class="d-grid gap-2">
            <button class="btn btn-primary" type="submit"><i class="bi bi-magic me-1"></i>Generate Dummy Data</button>
          </form>
          <form method="post" action="{{ url_for('do_upload') }}" enctype="multipart/form-data" class="d-grid gap-2 mt-2">
            <input type="file" class="form-control" name="file" accept=".csv">
            <button class="btn btn-outline-primary" type="submit"><i class="bi bi-upload me-1"></i>Upload CSV</button>
          </form>
          <hr>
          <form method="post" action="{{ url_for('do_train') }}" class="d-grid gap-2">
            <button class="btn btn-success" type="submit"><i class="bi bi-cpu me-1"></i>Train + Tune</button>
          </form>
          <form method="post" action="{{ url_for('do_score') }}" class="d-grid gap-2 mt-2">
            <button class="btn btn-warning" type="submit"><i class="bi bi-bullseye me-1"></i>Score & Bucket</button>
          </form>
          <a class="btn btn-outline-secondary mt-2 w-100" href="{{ url_for('download_scored') }}"><i class="bi bi-download me-1"></i>Download Scored CSV</a>
        </div>
      </div>
    </div>

    <div class="col-lg-8">
      <ul class="nav nav-tabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#data" type="button">Data & Target</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#trainperf" type="button">Performance (Deciles)</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#explain" type="button">Explainability</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#scores" type="button">Predictions & Leads</button></li>
      </ul>

      <div class="tab-content pt-3">
        <!-- Data -->
        <div class="tab-pane fade show active" id="data">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center">
                <h5 class="mb-0"><i class="bi bi-table me-1"></i>Sample (Top 10)</h5>
                <div><span class="badge text-bg-light">Rows: {{ rows }}</span>
                     <span class="badge text-bg-light">Cols: {{ cols }}</span></div>
              </div>
              {% if sample_html %}
                <div class="table-responsive mt-3">{{ sample_html|safe }}</div>
              {% else %}
                <p class="text-muted mt-3">Click “Generate Dummy Data” or upload a CSV.</p>
              {% endif %}
            </div>
          </div>
          {% if leakage_text %}
          <div class="card card-soft">
            <div class="card-body">
              <h6 class="mb-1"><i class="bi bi-shield-lock"></i> Target leakage check</h6>
              <p class="small text-muted">{{ leakage_text }}</p>
              <div class="table-responsive">{{ leakage_table|safe }}</div>
            </div>
          </div>
          {% endif %}
        </div>

        <!-- Training & Performance (deciles only) -->
        <div class="tab-pane fade" id="trainperf"> 
          <div class="card card-soft mb-3">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-speedometer2"></i> Test — Performance by Top-% (Deciles)</h5>
              {% if perf %}
                <div class="table-responsive">{{ perf.topk_test_html|safe }}</div>
                <div class="small text-muted mt-1">Base rate (test): {{ '%.3f'|format(perf.test_base_rate) }} | N={{ perf.yte_n }}</div>
              {% else %}
                <p class="text-muted">Train to view performance tables.</p>
              {% endif %}
            </div>
          </div>

          <div class="card card-soft">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-collection"></i> Train — Performance by Top-% (Deciles)</h5>
              {% if perf %}
                <div class="table-responsive">{{ perf.topk_train_html|safe }}</div>
                <div class="small text-muted mt-1">Base rate (train): {{ '%.3f'|format(perf.train_base_rate) }} | N={{ perf.ytr_n }}</div>
              {% else %}
                <p class="text-muted">Train to view performance tables.</p>
              {% endif %}
            </div>
          </div>
        </div>

        <!-- Explainability -->
        <div class="tab-pane fade" id="explain">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-stars"></i> Global Feature Importance (CatBoost)</h5>
              <div class="table-responsive">{{ imp_html|safe }}</div>
              <h6 class="mt-3">Permutation Importance (Accuracy)</h6>
              <div class="table-responsive">{{ perm_html|safe }}</div>
            </div>
          </div>
          <div class="card card-soft">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-chat-right-text"></i> Auto-Narrative</h5>
              <div class="small">{{ narrative_html|safe }}</div>
            </div>
          </div>
        </div>

        <!-- Predictions & Leads -->
        <div class="tab-pane fade" id="scores">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-bar-chart-steps"></i> Bucket Summary</h5>
              <div class="row g-3">
                {% for label in ['Very High','High','Moderate','Low','No Risk'] %}
                <div class="col-6 col-md">
                  <div class="p-2 border rounded text-center">
                    {{ label }}<br><b>{{ counts.get(label,0) }}</b>
                  </div>
                </div>
                {% endfor %}
              </div>
            </div>
          </div>
          <div class="card card-soft">
            <div class="card-body">
              <h5 class="mb-2"><i class="bi bi-person-lines-fill"></i> Top 20 Predictions</h5>
              {% if scored_html %}
                <div class="table-responsive">{{ scored_html|safe }}</div>
              {% else %}
                <p class="text-muted">Click “Score & Bucket”.</p>
              {% endif %}
              <a class="btn btn-outline-secondary mt-2" href="{{ url_for('download_scored') }}"><i class="bi bi-download me-1"></i>Download Scored CSV</a>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ---------- Entrypoint ----------
if __name__ == "__main__":
    port = _free_port(5008)
    url = f"http://127.0.0.1:{port}"
    print("\n===========================================")
    print(" Risk Insights Studio is starting")
    print(f" Visit {url}")
    print(" If nothing opens, copy-paste the URL above")
    print("===========================================\n")
    threading.Timer(1.0, lambda: webbrowser.open(url, new=2)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
