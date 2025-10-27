from __future__ import annotations
import io, webbrowser, threading
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template_string, flash, send_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

app = Flask(__name__)
app.secret_key = "change-me"

# State
DF: pd.DataFrame | None = None
MODEL = None
SCORED: pd.DataFrame | None = None
INFO: dict = {}

FEATURES = [
    "utilisation", "dpd_days", "cash_credit_ratio", "cash_debit_ratio",
    "inbound_cheque_bounce_count", "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count", "outbound_cheque_bounce_amt",
    "total_amt_credit", "total_amt_debit", "no_of_banks"
]

# ----------------------------
# Data generation (fixed)
# ----------------------------
def generate_dummy(n: int = 600) -> pd.DataFrame:
    """
    Generates a reproducible dataset with BOTH classes present.
    We derive a continuous risk score and classify by a quantile
    threshold (30% positives). If still single-class due to
    edge cases, we flip a small slice to guarantee both labels.
    """
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

    # Continuous risk score (0..1)
    prob = (
        0.25*(util/20) + 0.15*(dpd/365) + 0.15*cash_c + 0.15*cash_d
        + 0.10*(inb_cnt/30) + 0.10*(out_cnt/30)
        + 0.10*(np.maximum(inb_amt, out_amt)/5e5)
    )
    prob = np.clip(prob, 0, 1)

    # Turn into binary target using a quantile threshold => approx 30% positives
    thresh = np.quantile(prob, 0.70)
    y = (prob >= thresh).astype(np.int8)

    # Safety net: guarantee at least one of each class
    unique = np.unique(y)
    if unique.size < 2:
        # force top 30 samples to 1 and bottom 30 to 0
        order = np.argsort(prob)
        y[:] = 0
        y[order[-30:]] = 1

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

# ----------------------------
# Helpers
# ----------------------------
def leakage_report(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    corr = df.drop(columns=["customer_no"]).corr(numeric_only=True)["risky"] \
             .sort_values(key=lambda s: s.abs(), ascending=False)
    red_flags = [c for c, v in corr.items() if c != "risky" and abs(v) >= 0.90]
    txt = "No red-flags (|corr with target| ≥ 0.90)." if not red_flags \
          else "Potential leakage in: " + ", ".join(red_flags)
    return txt, corr.to_frame("corr_with_target")

def _check_classes(y: pd.Series) -> bool:
    """Return True if at least two classes exist."""
    return pd.Series(y).nunique() >= 2

def train_and_tune(df: pd.DataFrame):
    X = df[FEATURES].astype("float32")
    y = df["risky"].astype("int8")

    # Guard: if uploaded CSV has single class, synthesize a target by quantile
    if not _check_classes(y):
        # Use a proxy score and split at 70th percentile to form two classes
        proxy = (
            0.25*(df["utilisation"]/20) + 0.15*(df["dpd_days"]/365)
            + 0.15*df["cash_credit_ratio"] + 0.15*df["cash_debit_ratio"]
            + 0.10*(df["inbound_cheque_bounce_count"]/30)
            + 0.10*(df["outbound_cheque_bounce_count"]/30)
        )
        thresh = np.nanquantile(proxy, 0.70)
        y = (proxy >= thresh).astype("int8")
        flash("Uploaded data had one class. Created a balanced target via proxy score (70th percentile).", "warning")

    # Final safety
    if not _check_classes(y):
        # As a last resort flip the top 10 to class 1
        idx = np.argsort(X["utilisation"].to_numpy())
        y[:] = 0
        y.iloc[idx[-10:]] = 1
        flash("Target was still single-class; forced minimal positives for demo training.", "warning")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(solver="liblinear", max_iter=200, tol=1e-3))
    ])
    grid = {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]}
    gs = GridSearchCV(pipe, grid, cv=3, n_jobs=None)
    gs.fit(Xtr, ytr)

    best = gs.best_estimator_
    acc = best.score(Xte, yte)
    rep = classification_report(yte, best.predict(Xte), zero_division=0)
    cm = confusion_matrix(yte, best.predict(Xte))
    return best, {
        "best_C": float(gs.best_params_["clf__C"]),
        "best_penalty": gs.best_params_["clf__penalty"],
        "cv_score": float(gs.best_score_),
        "test_acc": float(acc),
        "report": rep,
        "cm": cm.tolist(),
    }

def importance_tables(model, df: pd.DataFrame):
    coef = model.named_steps["clf"].coef_.ravel()
    imp = pd.DataFrame({
        "feature": FEATURES,
        "coef": coef,
        "abs_importance": np.abs(coef)
    }).sort_values("abs_importance", ascending=False)

    X = df[FEATURES].astype("float32")
    y = df["risky"].astype("int8")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pim = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0)
    perm = pd.DataFrame({"feature": FEATURES, "perm_importance": pim.importances_mean}) \
                .sort_values("perm_importance", ascending=False)
    return imp, perm

def bucket(p: float) -> str:
    if p >= 0.90: return "Very High"
    if p >= 0.80: return "High"
    if p >= 0.70: return "Moderate"
    if p >= 0.60: return "Low"
    return "No Risk"

def score_all(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype("float32")
    proba = model.predict_proba(X)[:, 1]
    out = df[["customer_no"]].copy()
    out["prob"] = np.round(proba, 4)
    out["bucket"] = [bucket(p) for p in proba]

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    Xs = scaler.transform(X)
    contrib = Xs * clf.coef_[0]
    top_idx = np.argsort(-np.abs(contrib), axis=1)[:, :3]
    out["reasons"] = [", ".join(FEATURES[j] for j in top_idx[i]) for i in range(len(out))]

    return out.sort_values("prob", ascending=False)

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    rows, cols = (DF.shape if isinstance(DF, pd.DataFrame) else (0, 0))
    sample_html = DF.head(10).to_html(classes="table table-sm table-striped", index=False, border=0) \
                  if isinstance(DF, pd.DataFrame) else ""
    perf = INFO if INFO else None
    scored_html = SCORED.head(20).to_html(classes="table table-sm table-striped", index=False, border=0) \
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
    global DF, MODEL, INFO
    if DF is None:
        flash("Load dummy data or upload CSV first.", "warning")
        return redirect(url_for("home"))

    leak_txt, leak_corr_df = leakage_report(DF)
    MODEL, metrics = train_and_tune(DF)

    imp_df, perm_df = importance_tables(MODEL, DF)
    narrative_html = narrative_from_importance(imp_df)

    INFO = dict(metrics)
    INFO["imp_html"] = imp_df.to_html(classes="table table-sm table-striped", index=False, border=0)
    INFO["perm_html"] = perm_df.to_html(classes="table table-sm table-striped", index=False, border=0)
    INFO["narrative_html"] = narrative_html
    INFO["leakage_text"] = leak_txt
    INFO["leakage_table"] = leak_corr_df.to_html(classes="table table-sm table-striped", border=0)

    flash("Model trained & tuned.", "success")
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
    buf = io.StringIO()
    SCORED.to_csv(buf, index=False); buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="scored_portfolio.csv")

# ----------------------------
# HTML template
# ----------------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Risk Insights Studio — Flask (Fixed Dummy)</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{background:#f8fafc;color:#0f172a}
    .card-soft{border:1px solid #edf1f7;box-shadow:0 6px 18px rgba(16,24,40,.06);background:#fff;border-radius:18px}
    pre.console{background:#0b1020;color:#d1e7ff;padding:12px;border-radius:12px;white-space:pre-wrap}
  </style>
</head>
<body class="py-4">
<div class="container">
  <h3 class="mb-3">Risk Insights Studio — Flask (No logos)</h3>

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
          <h5>Actions</h5>
          <form method="post" action="{{ url_for('do_generate') }}" class="d-grid gap-2">
            <button class="btn btn-primary" type="submit">Generate Dummy Data</button>
          </form>
          <form method="post" action="{{ url_for('do_upload') }}" enctype="multipart/form-data" class="d-grid gap-2 mt-2">
            <input type="file" class="form-control" name="file" accept=".csv">
            <button class="btn btn-outline-primary" type="submit">Upload CSV</button>
          </form>
          <hr>
          <form method="post" action="{{ url_for('do_train') }}" class="d-grid gap-2">
            <button class="btn btn-success" type="submit">Train + Tune</button>
          </form>
          <form method="post" action="{{ url_for('do_score') }}" class="d-grid gap-2 mt-2">
            <button class="btn btn-warning" type="submit">Score & Bucket</button>
          </form>
          <a class="btn btn-outline-secondary mt-2 w-100" href="{{ url_for('download_scored') }}">Download Scored CSV</a>
          <hr>
          <div class="small text-muted">Pipeline: Train/Test Split • Standard Scaler • Logistic Regression (liblinear) • GridSearchCV • Iterative training (max_iter=200).</div>
        </div>
      </div>
    </div>

    <div class="col-lg-8">
      <ul class="nav nav-tabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#data" type="button">Data & Target</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#train" type="button">Training & Tuning</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#perf" type="button">Model Performance</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#explain" type="button">Explainability</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#scores" type="button">Predictions & Leads</button></li>
      </ul>

      <div class="tab-content pt-3">
        <div class="tab-pane fade show active" id="data">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <div class="d-flex justify-content-between">
                <h5 class="mb-0">Sample (Top 10)</h5>
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
              <h6 class="mb-1">Target leakage check</h6>
              <p class="small">{{ leakage_text }}</p>
              <div class="table-responsive">{{ leakage_table|safe }}</div>
            </div>
          </div>
          {% endif %}
        </div>

        <div class="tab-pane fade" id="train">
          <div class="card card-soft">
            <div class="card-body">
              {% if perf %}
              <div class="row g-3">
                <div class="col"><div class="p-2 border rounded text-center">Best C<br><b>{{ perf.best_C }}</b></div></div>
                <div class="col"><div class="p-2 border rounded text-center">Best Penalty<br><b>{{ perf.best_penalty }}</b></div></div>
                <div class="col"><div class="p-2 border rounded text-center">CV Score<br><b>{{ '%.3f'|format(perf.cv_score) }}</b></div></div>
                <div class="col"><div class="p-2 border rounded text-center">Test Acc<br><b>{{ '%.3f'|format(perf.test_acc) }}</b></div></div>
              </div>
              {% else %}
                <p class="text-muted">Train to see best params and scores.</p>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="tab-pane fade" id="perf">
          <div class="card card-soft">
            <div class="card-body">
              {% if perf %}
                <h5>Classification Report</h5>
                <pre class="console">{{ perf.report }}</pre>
                <h6>Confusion Matrix</h6>
                <table class="table table-sm table-bordered w-auto">
                  {% for row in perf.cm %}
                    <tr>{% for c in row %}<td>{{ c }}</td>{% endfor %}</tr>
                  {% endfor %}
                </table>
              {% else %}
                <p class="text-muted">Train first to view performance.</p>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="tab-pane fade" id="explain">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <h5 class="mb-2">Variable Importance (Standardized Coefficients)</h5>
              <div class="table-responsive">{{ imp_html|safe }}</div>
              <h6 class="mt-3">Permutation Importance</h6>
              <div class="table-responsive">{{ perm_html|safe }}</div>
            </div>
          </div>
          <div class="card card-soft">
            <div class="card-body">
              <h5 class="mb-2">Auto-Narrative</h5>
              <div class="small">{{ narrative_html|safe }}</div>
            </div>
          </div>
        </div>

        <div class="tab-pane fade" id="scores">
          <div class="card card-soft mb-3">
            <div class="card-body">
              <h5>Bucket Summary</h5>
              <div class="row g-3">
                {% for label in ['Very High','High','Moderate','Low','No Risk'] %}
                <div class="col-6 col-md">
                  <div class="p-2 border rounded text-center">{{ label }}<br><b>{{ counts.get(label,0) }}</b></div>
                </div>
                {% endfor %}
              </div>
            </div>
          </div>
          <div class="card card-soft">
            <div class="card-body">
              <h5>Top 20 Predictions</h5>
              {% if scored_html %}
                <div class="table-responsive">{{ scored_html|safe }}</div>
              {% else %}
                <p class="text-muted">Click “Score & Bucket”.</p>
              {% endif %}
              <a class="btn btn-outline-secondary mt-2" href="{{ url_for('download_scored') }}">Download Scored CSV</a>
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

def narrative_from_importance(imp_df: pd.DataFrame) -> str:
    top = imp_df.head(5).reset_index(drop=True)
    bullets = [f"{i+1}. {row.feature} (impact {row.abs_importance:.2f})" for i, row in top.iterrows()]
    return (
        "<p><b>Executive summary:</b> Model risk is driven primarily by:</p>"
        "<ul><li>" + "</li><li>".join(bullets) + "</li></ul>"
        "<p>Higher values on the leading drivers typically raise risk probability; "
        "focus mitigation and outreach on these variables.</p>"
    )

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"\n🚀 TMU Flask app ready — visit {url}\n")
    # No auto-open here to avoid issues on restricted machines; open manually if needed.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
