"""Credit Scoring Agent (UI + API)

This file provides an agentic credit-scoring flow with:
- Single scoring (sync) with explanation
- Batch scoring (async jobs) with audit trail
- UI in the same style/format as payment_screening.py

Model behavior:
- Tries to load a pickled model from `credit_risk_model.pkl` if available.
- If the pickle can't be loaded (e.g., CatBoost not installed), uses an embedded
	deterministic scorer derived from `train_and_save_model.py`.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for

try:
		import pandas as pd
except Exception:  # pragma: no cover
		pd = None


app = Flask(__name__)

# -----------------------------
# Model + scoring primitives
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"

FEATURES: List[str] = [
		"utilisation",
		"dpd_days",
		"cash_credit_ratio",
		"cash_debit_ratio",
		"inbound_cheque_bounce_count",
		"inbound_cheque_bounce_amt",
		"outbound_cheque_bounce_count",
		"outbound_cheque_bounce_amt",
		"total_amt_credit",
		"total_amt_debit",
		"no_of_banks",
]

DECISION_THRESHOLDS = {
		"ACCEPT_MAX": 0.50,
		"CONDITIONAL_ACCEPT_MAX": 0.70,
		"REVIEW_MAX": 0.80,
}

RISK_BUCKET_THRESHOLDS = {
		"Very High": 0.90,
		"High": 0.80,
		"Moderate": 0.70,
		"Low": 0.50,
		"No Risk": 0.00,
}


MODEL: Any = None
MODEL_LOCK = threading.Lock()


def _utc_now_iso() -> str:
		return datetime.now(timezone.utc).isoformat()


def load_model() -> None:
		global MODEL
		with MODEL_LOCK:
				if MODEL is not None:
						return

				if not MODEL_PATH.exists():
						MODEL = None
						print(f"⚠ Warning: {MODEL_PATH} not found, using embedded scorer.")
						return

				try:
						with open(MODEL_PATH, "rb") as handle:
								MODEL = pickle.load(handle)
						print(f"✓ Model loaded from {MODEL_PATH}")
				except Exception as exc:
						MODEL = None
						print(
								f"⚠ Warning: Failed to load {MODEL_PATH} ({type(exc).__name__}: {exc}). "
								"Using embedded scorer."
						)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
		return float(max(lo, min(hi, value)))


def validate_payload(item: Dict[str, Any]) -> Tuple[bool, str]:
		missing = [feature for feature in FEATURES if feature not in item]
		if missing:
				return False, f"Missing required features: {', '.join(missing)}"

		for feature in FEATURES:
				try:
						value = float(item[feature])
				except (TypeError, ValueError):
						return False, f"Feature '{feature}' must be numeric"
				if value != value or value in (float("inf"), float("-inf")):
						return False, f"Feature '{feature}' must be finite"

		utilisation = float(item["utilisation"])
		dpd_days = float(item["dpd_days"])
		no_of_banks = float(item["no_of_banks"])

		if utilisation < 0 or utilisation > 100:
				return False, "utilisation must be in range 0-100"
		if dpd_days < 0:
				return False, "dpd_days cannot be negative"
		if no_of_banks < 0:
				return False, "no_of_banks cannot be negative"

		return True, ""


def normalize_features(item: Dict[str, Any]) -> Dict[str, float]:
		return {feature: float(item[feature]) for feature in FEATURES}


def embedded_probability(features: Dict[str, float]) -> float:
		"""Embedded scorer derived from train_and_save_model.generate_dummy_data()."""

		utilisation = float(features["utilisation"])
		dpd_days = float(features["dpd_days"])
		cash_credit_ratio = float(features["cash_credit_ratio"])
		cash_debit_ratio = float(features["cash_debit_ratio"])
		inbound_cnt = float(features["inbound_cheque_bounce_count"])
		outbound_cnt = float(features["outbound_cheque_bounce_count"])
		inbound_amt = float(features["inbound_cheque_bounce_amt"])
		outbound_amt = float(features["outbound_cheque_bounce_amt"])

		prob = (
				0.25 * _clamp(utilisation / 20.0)
				+ 0.15 * _clamp(dpd_days / 365.0)
				+ 0.15 * _clamp(cash_credit_ratio)
				+ 0.15 * _clamp(cash_debit_ratio)
				+ 0.10 * _clamp(inbound_cnt / 30.0)
				+ 0.10 * _clamp(outbound_cnt / 30.0)
				+ 0.10 * _clamp(max(inbound_amt, outbound_amt) / 500_000.0)
		)
		return _clamp(prob)


def predict_probability(features: Dict[str, float]) -> float:
		if MODEL is None:
				return embedded_probability(features)

		# If pandas isn't available, avoid breaking runtime.
		if pd is None:
				return embedded_probability(features)

		data = pd.DataFrame([features])
		probability = float(MODEL.predict_proba(data)[:, 1][0])
		return _clamp(probability)


def get_risk_bucket(probability: float) -> str:
		for bucket, threshold in RISK_BUCKET_THRESHOLDS.items():
				if probability >= threshold:
						return bucket
		return "No Risk"


def get_decision(probability: float) -> Tuple[str, str]:
		if probability < DECISION_THRESHOLDS["ACCEPT_MAX"]:
				return "ACCEPT", f"probability < {DECISION_THRESHOLDS['ACCEPT_MAX']:.2f}"
		if probability < DECISION_THRESHOLDS["CONDITIONAL_ACCEPT_MAX"]:
				return "CONDITIONAL_ACCEPT", f"{DECISION_THRESHOLDS['ACCEPT_MAX']:.2f} <= probability < {DECISION_THRESHOLDS['CONDITIONAL_ACCEPT_MAX']:.2f}"
		if probability < DECISION_THRESHOLDS["REVIEW_MAX"]:
				return "REVIEW", f"{DECISION_THRESHOLDS['CONDITIONAL_ACCEPT_MAX']:.2f} <= probability < {DECISION_THRESHOLDS['REVIEW_MAX']:.2f}"
		return "REJECT", f"probability >= {DECISION_THRESHOLDS['REVIEW_MAX']:.2f}"


def top_risk_drivers(features: Dict[str, float]) -> List[Dict[str, Any]]:
		"""Top drivers for embedded model (linear weights)."""

		util_n = _clamp(float(features["utilisation"]) / 20.0)
		dpd_n = _clamp(float(features["dpd_days"]) / 365.0)
		cash_c = _clamp(float(features["cash_credit_ratio"]))
		cash_d = _clamp(float(features["cash_debit_ratio"]))
		inb_n = _clamp(float(features["inbound_cheque_bounce_count"]) / 30.0)
		out_n = _clamp(float(features["outbound_cheque_bounce_count"]) / 30.0)
		bounce_amt_n = _clamp(max(float(features["inbound_cheque_bounce_amt"]), float(features["outbound_cheque_bounce_amt"])) / 500_000.0)

		impacts: List[Tuple[str, float, float, str]] = [
				("utilisation", 0.25, util_n, f"Utilisation at {features['utilisation']:.1f}%"),
				("dpd_days", 0.15, dpd_n, f"DPD days at {features['dpd_days']:.0f}"),
				("cash_credit_ratio", 0.15, cash_c, f"Cash credit ratio {features['cash_credit_ratio']:.2f}"),
				("cash_debit_ratio", 0.15, cash_d, f"Cash debit ratio {features['cash_debit_ratio']:.2f}"),
				("inbound_cheque_bounce_count", 0.10, inb_n, "Inbound cheque bounce count"),
				("outbound_cheque_bounce_count", 0.10, out_n, "Outbound cheque bounce count"),
				("bounce_amount", 0.10, bounce_amt_n, "Cheque bounce amount exposure"),
		]

		scored = []
		for name, weight, normalized_value, summary in impacts:
				scored.append((name, float(weight * normalized_value), summary))

		scored_sorted = sorted(scored, key=lambda row: abs(row[1]), reverse=True)[:3]
		out: List[Dict[str, Any]] = []
		for name, impact, summary in scored_sorted:
				out.append({"factor": name, "impact": round(float(impact), 4), "summary": summary})
		return out


def build_explanation(customer_id: str, probability: float, bucket: str, decision: str, policy: str, drivers: List[Dict[str, Any]]) -> str:
		lines: List[str] = []
		lines.append(f"Customer: {customer_id}")
		lines.append(f"Probability: {probability:.4f} ({probability * 100:.2f}%)")
		lines.append(f"Bucket: {bucket}")
		lines.append(f"Decision: {decision}")
		lines.append(f"Policy: {policy}")
		lines.append("")
		lines.append("Top drivers:")
		for d in drivers:
				lines.append(f"- {d.get('factor')}: impact={d.get('impact')}  ({d.get('summary')})")
		lines.append("")
		lines.append(
				"Decision rules (probability of default): "
				f"< {DECISION_THRESHOLDS['ACCEPT_MAX']:.2f} => ACCEPT; "
				f"< {DECISION_THRESHOLDS['CONDITIONAL_ACCEPT_MAX']:.2f} => CONDITIONAL_ACCEPT; "
				f"< {DECISION_THRESHOLDS['REVIEW_MAX']:.2f} => REVIEW; "
				f">= {DECISION_THRESHOLDS['REVIEW_MAX']:.2f} => REJECT"
		)
		return "\n".join(lines)


# -----------------------------
# Jobs + audit
# -----------------------------
MAX_RECENT_JOBS = 50
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _append_audit(job: Dict[str, Any], step: str, message: str, item_index: Optional[int] = None, data: Optional[Dict[str, Any]] = None) -> None:
		job["audit"].append(
				{
						"ts": _utc_now_iso(),
						"job_id": job["job_id"],
						"item_index": item_index,
						"step": step,
						"message": message,
						"data": data or {},
				}
		)


def _new_job(items: List[Dict[str, Any]]) -> Dict[str, Any]:
		job_id = uuid.uuid4().hex
		return {
				"job_id": job_id,
				"status": "QUEUED",
				"created_at": _utc_now_iso(),
				"started_at": None,
				"finished_at": None,
				"error": None,
				"total_items": len(items),
				"completed_items": 0,
				"accepted": 0,
				"conditional_accept": 0,
				"review": 0,
				"rejected": 0,
				"avg_score": 0.0,
				"max_score": 0.0,
				"items": items,
				"results": [],
				"audit": [],
		}


def _kpis_snapshot() -> Dict[str, Any]:
		with JOBS_LOCK:
				jobs = list(JOBS.values())

		jobs_total = len(jobs)
		jobs_queued = sum(1 for j in jobs if j.get("status") == "QUEUED")
		jobs_running = sum(1 for j in jobs if j.get("status") == "RUNNING")
		jobs_failed = sum(1 for j in jobs if j.get("status") == "FAILED")

		items_total = sum(int(j.get("total_items") or 0) for j in jobs)
		items_completed = sum(int(j.get("completed_items") or 0) for j in jobs)

		accepted = sum(int(j.get("accepted") or 0) for j in jobs)
		rejected = sum(int(j.get("rejected") or 0) for j in jobs)

		return {
				"jobs_total": jobs_total,
				"jobs_queued": jobs_queued,
				"jobs_running": jobs_running,
				"jobs_failed": jobs_failed,
				"items_total": items_total,
				"items_completed": items_completed,
				"decisions_accept": accepted,
				"decisions_reject": rejected,
				"reject_rate": (rejected / items_completed) if items_completed else 0.0,
		}


def _parse_batch_json(raw: bytes) -> List[Dict[str, Any]]:
		obj = json.loads(raw.decode("utf-8"))

		# Accept:
		# 1) list[payload]
		# 2) {"items": [payload,...]}
		# 3) {"customers": [payload,...]} (used by test_payloads/batch_*.json)
		if isinstance(obj, list):
				return [dict(x) for x in obj if isinstance(x, dict)]

		if isinstance(obj, dict) and isinstance(obj.get("items"), list):
				return [dict(x) for x in obj["items"] if isinstance(x, dict)]

		if isinstance(obj, dict) and isinstance(obj.get("customers"), list):
				return [dict(x) for x in obj["customers"] if isinstance(x, dict)]

		raise ValueError("Unsupported batch JSON format. Expected list, {items:[...]}, or {customers:[...]}")


def _process_job(job_id: str) -> None:
		load_model()

		with JOBS_LOCK:
				job = JOBS.get(job_id)
				if not job:
						return
				job["status"] = "RUNNING"
				job["started_at"] = _utc_now_iso()
				_append_audit(job, "job_start", "Job started")

		try:
				scores: List[float] = []
				for idx, payload in enumerate(job.get("items") or []):
						audit_sink: List[Dict[str, Any]] = []
						result = run_agentic_scoring(payload, job_id=job_id, item_index=idx, audit_sink=audit_sink)

						with JOBS_LOCK:
								job = JOBS.get(job_id)
								if not job:
										return

								job["results"].append({"index": idx, "result": result})
								job["completed_items"] = int(job.get("completed_items") or 0) + 1
								job["audit"].extend(audit_sink)

								status = (result.get("decision") or "").upper()
								if status == "ACCEPT":
										job["accepted"] += 1
								elif status == "CONDITIONAL_ACCEPT":
										job["conditional_accept"] += 1
								elif status == "REVIEW":
										job["review"] += 1
								elif status == "REJECT":
										job["rejected"] += 1

								probability = result.get("risk_score", {}).get("probability")
								if isinstance(probability, (int, float)):
										scores.append(float(probability))
										job["avg_score"] = float(sum(scores) / len(scores))
										job["max_score"] = float(max(scores))

				with JOBS_LOCK:
						job = JOBS.get(job_id)
						if job:
								job["status"] = "DONE"
								job["finished_at"] = _utc_now_iso()
								_append_audit(job, "job_done", "Job finished")

		except Exception as exc:
				with JOBS_LOCK:
						job = JOBS.get(job_id)
						if job:
								job["status"] = "FAILED"
								job["finished_at"] = _utc_now_iso()
								job["error"] = str(exc)
								_append_audit(job, "job_failed", "Job failed", data={"error": str(exc)})


# -----------------------------
# Agentic flow
# -----------------------------
def run_agentic_scoring(payload: Dict[str, Any], job_id: str, item_index: int, audit_sink: List[Dict[str, Any]]) -> Dict[str, Any]:
		def audit(step: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
				audit_sink.append(
						{
								"ts": _utc_now_iso(),
								"job_id": job_id,
								"item_index": item_index,
								"step": step,
								"message": message,
								"data": data or {},
						}
				)

		audit("flow", "Starting agentic credit scoring")
		is_valid, error_msg = validate_payload(payload)
		if not is_valid:
				audit("validate", "Payload invalid", {"error": error_msg})
				return {
						"customer_id": str(payload.get("customer_no") or payload.get("customer_id") or f"item_{item_index}"),
						"status": "ERROR",
						"error": error_msg,
						"decision": "ERROR",
						"policy": "validation_failed",
						"risk_score": {"probability": 0.0, "probability_percentage": "0.00%", "bucket": "No Risk"},
						"risk_drivers": [],
						"explanation": f"Validation failed: {error_msg}",
				}

		audit("normalize", "Normalizing features")
		features = normalize_features(payload)
		customer_id = str(payload.get("customer_no") or payload.get("customer_id") or f"item_{item_index}")

		audit("score", "Computing probability")
		probability = predict_probability(features)
		audit("score", "Probability computed", {"probability": round(probability, 6)})

		bucket = get_risk_bucket(probability)
		audit("bucket", "Bucket assigned", {"bucket": bucket})

		decision, policy = get_decision(probability)
		audit("decision", "Decision made", {"decision": decision, "policy": policy})

		drivers = top_risk_drivers(features)
		explanation = build_explanation(customer_id, probability, bucket, decision, policy, drivers)
		audit("explain", "Explanation generated")

		audit("flow", "Completed agentic credit scoring")
		return {
				"customer_id": customer_id,
				"timestamp": _utc_now_iso(),
				"risk_score": {
						"probability": round(probability, 4),
						"probability_percentage": f"{probability * 100:.2f}%",
						"bucket": bucket,
				},
				"decision": decision,
				"policy": policy,
				"risk_drivers": drivers,
				"model": "pkl" if MODEL is not None else "embedded",
				"explanation": explanation,
				"status": "OK",
		}


# -----------------------------
# Scenario loading for dropdown
# -----------------------------
def _load_test_scenarios() -> Dict[str, Dict[str, Any]]:
		payload_dir = BASE_DIR / "test_payloads"
		out: Dict[str, Dict[str, Any]] = {}
		if not payload_dir.is_dir():
				return out

		for path in sorted(payload_dir.glob("*.json")):
				name = path.stem
				if name.startswith("batch_"):
						continue
				try:
						payload = json.loads(path.read_text(encoding="utf-8"))
				except Exception:
						continue
				if not isinstance(payload, dict):
						continue
				# Only show payloads that match the single-item schema.
				if any(f not in payload for f in FEATURES):
						continue
				out[name] = payload

		return out


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, str]:
		out: Dict[str, str] = {}
		out["customer_no"] = str(payload.get("customer_no") or "")
		for f in FEATURES:
				v = payload.get(f, "")
				out[f] = "" if v is None else str(v)
		return out


# -----------------------------
# UI templates
# -----------------------------
TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
	<meta charset=\"utf-8\">
	<title>Credit Scoring (Agentic)</title>
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
	<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
	<style>
		:root {
			--bs-body-bg: #0f172a;
			--bs-body-color: #e2e8f0;
			--bs-secondary-color: #94a3b8;
			--bs-border-color: rgba(148, 163, 184, .25);

			--bs-nav-link-color: var(--bs-body-color);
			--bs-nav-link-hover-color: #ffffff;
			--bs-nav-tabs-border-color: var(--bs-border-color);
			--bs-nav-tabs-link-active-color: var(--bs-body-color);
			--bs-nav-tabs-link-active-bg: var(--bs-body-bg);
			--bs-nav-tabs-link-active-border-color: var(--bs-border-color);
		}

		body { background: var(--bs-body-bg); color: var(--bs-body-color); }
		.card { border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,.3); }
		.muted { color: #94a3b8; }
		.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
		.kv { display: grid; grid-template-columns: 160px 1fr; gap: .25rem .75rem; }
		.kv .k { color: var(--bs-secondary-color); }
		pre {
			white-space: pre-wrap;
			overflow-wrap: anywhere;
			word-break: break-word;
			line-height: 1.35;
			font-size: 0.92rem;
			color: var(--bs-body-color);
			border-color: var(--bs-border-color) !important;
		}
		pre.explain { max-height: 520px; overflow: auto; }

		.nav-tabs { border-bottom-color: var(--bs-border-color); gap: .25rem; }
		.nav-tabs .nav-link {
			color: var(--bs-body-color) !important;
			font-weight: 700;
			font-size: 1.02rem;
			letter-spacing: .2px;
			padding: .65rem .95rem;
			border-color: transparent;
			border-top-left-radius: .75rem;
			border-top-right-radius: .75rem;
		}
		.nav-tabs .nav-link:hover {
			color: #ffffff !important;
			border-color: var(--bs-border-color) var(--bs-border-color) transparent;
			background: rgba(226, 232, 240, .06);
		}
		.nav-tabs .nav-link.active {
			color: var(--bs-body-color) !important;
			background-color: var(--bs-body-bg) !important;
			border-color: var(--bs-border-color) var(--bs-border-color) var(--bs-body-bg) !important;
		}
		.nav-tabs .nav-link:focus-visible {
			outline: none;
			box-shadow: 0 0 0 .2rem rgba(13, 202, 240, .25);
		}

		input[type=\"file\"].form-control {
			background-color: var(--bs-body-bg);
			color: var(--bs-body-color);
			border-color: var(--bs-border-color);
		}
		input[type=\"file\"].form-control::file-selector-button {
			margin-right: .75rem;
			background: rgba(226, 232, 240, .10);
			color: var(--bs-body-color);
			border: 1px solid var(--bs-border-color);
			border-radius: .5rem;
			padding: .45rem .75rem;
			font-weight: 700;
		}
		input[type=\"file\"].form-control:hover::file-selector-button {
			background: rgba(226, 232, 240, .16);
		}
	</style>
</head>
<body>
<div class=\"container py-4\">
	<div class=\"card border-0\">
		<div class=\"card-header bg-dark\">
			<div class=\"d-flex justify-content-between align-items-start\">
				<div>
					<h3 class=\"m-0\">Credit Risk Scoring (Agentic Flow)</h3>
					<div class=\"muted\">Sample data → Algorithms → Scoring → Automated decision → Explanation • Batch jobs + audit trail</div>
				</div>
				<div class=\"text-end\">
					<a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('index') }}\">Refresh</a>
				</div>
			</div>
		</div>

		<div class=\"card-body bg-dark\">
			<ul class=\"nav nav-tabs\" role=\"tablist\">
				<li class=\"nav-item\" role=\"presentation\">
					<button class=\"nav-link active\" data-bs-toggle=\"tab\" data-bs-target=\"#tab-main\" type=\"button\" role=\"tab\">Main (KPIs + Batch)</button>
				</li>
				<li class=\"nav-item\" role=\"presentation\">
					<button class=\"nav-link\" data-bs-toggle=\"tab\" data-bs-target=\"#tab-jobs\" type=\"button\" role=\"tab\">Jobs + Audit Trail</button>
				</li>
			</ul>

			<div class=\"tab-content pt-3\">
				<div class=\"tab-pane fade show active\" id=\"tab-main\" role=\"tabpanel\">
					<div class=\"row g-3\">
						<div class=\"col-md-3\">
							<div class=\"border rounded p-3\">
								<div class=\"muted\">Jobs</div>
								<div class=\"fs-4\">{{ kpis.jobs_total }}</div>
								<div class=\"muted\">Queued: {{ kpis.jobs_queued }} • Running: {{ kpis.jobs_running }} • Failed: {{ kpis.jobs_failed }}</div>
							</div>
						</div>
						<div class=\"col-md-3\">
							<div class=\"border rounded p-3\">
								<div class=\"muted\">Items Processed</div>
								<div class=\"fs-4\">{{ kpis.items_completed }} / {{ kpis.items_total }}</div>
								<div class=\"muted\">Across all batch jobs</div>
							</div>
						</div>
						<div class=\"col-md-3\">
							<div class=\"border rounded p-3\">
								<div class=\"muted\">Decisions</div>
								<div class=\"fs-4\">A: {{ kpis.decisions_accept }} • REJ: {{ kpis.decisions_reject }}</div>
								<div class=\"muted\">Reject rate: {{ (kpis.reject_rate * 100) | round(1) }}%</div>
							</div>
						</div>
						<div class=\"col-md-3\">
							<div class=\"border rounded p-3\">
								<div class=\"muted\">Policy</div>
								<div class=\"fs-4\">Reject: {{ reject_threshold }}</div>
								<div class=\"muted\">Accept: &lt; {{ accept_threshold }} • Review: &lt; {{ reject_threshold }}</div>
							</div>
						</div>
					</div>

					<hr class=\"border-secondary\" />

					<div class=\"row g-3\">
						<div class=\"col-lg-6\">
							<h5 class=\"text-info\">Single Application (Sync)</h5>

							<form method=\"get\" class=\"mb-3\">
								<label class=\"form-label\">Load sample scenario</label>
								<div class=\"input-group\">
									<select class=\"form-select\" name=\"scenario\" onchange=\"this.form.submit()\">
										<option value=\"\">-- Choose --</option>
										{% for s in scenarios %}
											<option value=\"{{ s }}\" {% if s == selected_scenario %}selected{% endif %}>{{ s }}</option>
										{% endfor %}
									</select>
									<button class=\"btn btn-outline-light\" type=\"submit\">Load</button>
								</div>
								<div class=\"muted mt-1\">Uses data from <span class=\"mono\">test_payloads/</span></div>
							</form>

							<form method=\"post\" action=\"{{ url_for('score_sync') }}\" class=\"row g-2\">
								<div class=\"col-12\"><div class=\"muted\">Customer</div></div>
								<div class=\"col-12\"><input class=\"form-control\" name=\"customer_no\" placeholder=\"customer_no\" value=\"{{ form.customer_no }}\"></div>

								<div class=\"col-12 pt-2\"><div class=\"muted\">Features</div></div>
								{% for f in features %}
									<div class=\"col-md-6\">
										<input class=\"form-control\" name=\"{{ f }}\" placeholder=\"{{ f }}\" value=\"{{ form[f] }}\" required>
									</div>
								{% endfor %}

								<div class=\"col-12 pt-2\">
									<button class=\"btn btn-info\" type=\"submit\">Run Agentic Scoring</button>
								</div>
							</form>

							{% if sync_result %}
								<hr class=\"border-secondary\" />
								<h6 class=\"text-warning\">Result</h6>

								{% set _dec = (sync_result.decision or '') %}
								{% set _dec_class = 'text-bg-secondary' %}
								{% if _dec == 'REJECT' %}{% set _dec_class = 'text-bg-danger' %}{% endif %}
								{% if _dec == 'REVIEW' %}{% set _dec_class = 'text-bg-warning text-dark' %}{% endif %}
								{% if _dec == 'CONDITIONAL_ACCEPT' %}{% set _dec_class = 'text-bg-info text-dark' %}{% endif %}
								{% if _dec == 'ACCEPT' %}{% set _dec_class = 'text-bg-success' %}{% endif %}

								<div class=\"border rounded p-3 bg-black\">
									<div class=\"d-flex flex-wrap gap-2 align-items-center\">
										<span class=\"badge {{ _dec_class }}\">{{ sync_result.decision }}</span>
										<span class=\"badge text-bg-dark\">{{ sync_result.policy }}</span>
										<span class=\"badge text-bg-dark\">Model: {{ sync_result.model }}</span>
									</div>

									{% set risk = sync_result.risk_score or {} %}
									<div class=\"kv mt-3\">
										<div class=\"k\">Probability</div>
										<div class=\"mono\">{{ risk.probability }} ({{ risk.probability_percentage }})</div>
										<div class=\"k\">Bucket</div>
										<div class=\"mono\">{{ risk.bucket }}</div>
									</div>
								</div>

								<div class=\"mt-3\">
									<h6 class=\"text-warning\">Explanation</h6>
									<pre class=\"explain mono border rounded p-3 bg-black\">{{ sync_result.explanation }}</pre>
								</div>
							{% endif %}
						</div>

						<div class=\"col-lg-6\">
							<h5 class=\"text-info\">Batch Scoring (Async Job)</h5>
							<div class=\"muted\">Upload JSON as: list[payload] OR {items:[...]} OR {customers:[...]}</div>

							<form method=\"post\" action=\"{{ url_for('submit_job') }}\" enctype=\"multipart/form-data\" class=\"mt-2\">
								<div class=\"mb-2\">
									<input class=\"form-control\" type=\"file\" name=\"batch_file\" accept=\"application/json\" required>
								</div>
								<button class=\"btn btn-outline-info\" type=\"submit\">Submit Batch Job</button>
							</form>

							{% if submit_msg %}
								<div class=\"alert alert-secondary mt-3\">{{ submit_msg }}</div>
							{% endif %}

							<hr class=\"border-secondary\" />
							<h6 class=\"text-warning\">Recent Jobs</h6>
							<div class=\"table-responsive\">
								<table class=\"table table-dark table-sm align-middle\">
									<thead>
										<tr>
											<th>Job</th>
											<th>Status</th>
											<th>Done</th>
											<th>A/REJ</th>
											<th>Avg</th>
											<th>Max</th>
											<th></th>
										</tr>
									</thead>
									<tbody>
										{% for j in jobs %}
											<tr>
												<td class=\"mono\">{{ j.job_id[:8] }}</td>
												<td>{{ j.status }}</td>
												<td>{{ j.completed_items }}/{{ j.total_items }}</td>
												<td>{{ j.accepted }}/{{ j.rejected }}</td>
												<td>{{ j.avg_score | round(3) }}</td>
												<td>{{ j.max_score | round(3) }}</td>
												<td><a class=\"btn btn-sm btn-outline-light\" href=\"{{ url_for('job_view', job_id=j.job_id) }}\">Open</a></td>
											</tr>
										{% endfor %}
										{% if not jobs %}
											<tr><td colspan=\"7\" class=\"muted\">No jobs yet.</td></tr>
										{% endif %}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				</div>

				<div class=\"tab-pane fade\" id=\"tab-jobs\" role=\"tabpanel\">
					<h5 class=\"text-info\">Job Status & Audit Trail</h5>
					<div class=\"muted\">Audit events are kept in memory for this demo.</div>

					<div class=\"mt-3\">
						<div class=\"table-responsive\">
							<table class=\"table table-dark table-sm align-middle\">
								<thead>
									<tr>
										<th>Job</th>
										<th>Status</th>
										<th>Created</th>
										<th>Started</th>
										<th>Finished</th>
										<th>Done</th>
										<th></th>
									</tr>
								</thead>
								<tbody>
									{% for j in jobs %}
										<tr>
											<td class=\"mono\">{{ j.job_id }}</td>
											<td>{{ j.status }}</td>
											<td class=\"mono\">{{ j.created_at }}</td>
											<td class=\"mono\">{{ j.started_at or '—' }}</td>
											<td class=\"mono\">{{ j.finished_at or '—' }}</td>
											<td>{{ j.completed_items }}/{{ j.total_items }}</td>
											<td><a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('job_view', job_id=j.job_id) }}\">View + Audit</a></td>
										</tr>
									{% endfor %}
									{% if not jobs %}
										<tr><td colspan=\"7\" class=\"muted\">No jobs yet.</td></tr>
									{% endif %}
								</tbody>
							</table>
						</div>
					</div>
				</div>

			</div>
		</div>
	</div>

	<div class=\"muted mt-3\">
		API: <span class=\"mono\">/api/jobs</span>, <span class=\"mono\">/api/jobs/&lt;job_id&gt;</span>, <span class=\"mono\">/api/jobs/&lt;job_id&gt;/audit</span>
	</div>
</div>

<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body>
</html>
"""


JOB_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
	<meta charset=\"utf-8\">
	<title>Job {{ job.job_id }}</title>
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
	<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
	<style>
		:root {
			--bs-body-bg: #0f172a;
			--bs-body-color: #e2e8f0;
			--bs-secondary-color: #94a3b8;
			--bs-border-color: rgba(148, 163, 184, .25);
		}
		body { background: var(--bs-body-bg); color: var(--bs-body-color); }
		.muted { color: #94a3b8; }
		.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
		pre {
			white-space: pre-wrap;
			overflow-wrap: anywhere;
			word-break: break-word;
			line-height: 1.35;
			font-size: 0.92rem;
			color: var(--bs-body-color);
			border-color: var(--bs-border-color) !important;
		}
		pre.explain { max-height: 520px; overflow: auto; }
	</style>
</head>
<body>
<div class=\"container py-4\">
	<div class=\"d-flex justify-content-between align-items-start\">
		<div>
			<h3 class=\"m-0\">Job</h3>
			<div class=\"mono\">{{ job.job_id }}</div>
			<div class=\"muted\">Status: {{ job.status }} • Done {{ job.completed_items }}/{{ job.total_items }}</div>
		</div>
		<div class=\"text-end\">
			<a class=\"btn btn-sm btn-outline-light\" href=\"{{ url_for('index') }}\">Back</a>
			<a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('job_view', job_id=job.job_id) }}\">Refresh</a>
		</div>
	</div>

	{% if job.error %}
		<div class=\"alert alert-danger mt-3\">{{ job.error }}</div>
	{% endif %}

	<hr class=\"border-secondary\" />
	<h5 class=\"text-info\">Results</h5>
	<div class=\"table-responsive\">
		<table class=\"table table-dark table-sm align-middle\">
			<thead>
				<tr>
					<th>#</th>
					<th>Customer</th>
					<th>Decision</th>
					<th>Bucket</th>
					<th>Probability</th>
					<th></th>
				</tr>
			</thead>
			<tbody>
				{% for r in job.results %}
					{% set res = r.result %}
					{% set risk = res.risk_score or {} %}
					<tr>
						<td>{{ r.index }}</td>
						<td class=\"mono\">{{ res.customer_id }}</td>
						<td>
							{% set _dec = (res.decision or '') %}
							{% set _dec_class = 'text-bg-secondary' %}
							{% if _dec == 'REJECT' %}{% set _dec_class = 'text-bg-danger' %}{% endif %}
							{% if _dec == 'REVIEW' %}{% set _dec_class = 'text-bg-warning text-dark' %}{% endif %}
							{% if _dec == 'CONDITIONAL_ACCEPT' %}{% set _dec_class = 'text-bg-info text-dark' %}{% endif %}
							{% if _dec == 'ACCEPT' %}{% set _dec_class = 'text-bg-success' %}{% endif %}
							<span class=\"badge {{ _dec_class }}\">{{ res.decision }}</span>
						</td>
						<td>{{ risk.bucket }}</td>
						<td class=\"mono\">{{ risk.probability }} ({{ risk.probability_percentage }})</td>
						<td>
							<details>
								<summary class=\"text-warning\">Explanation</summary>
								<pre class=\"explain mono border rounded p-2 bg-black\">{{ res.explanation }}</pre>
							</details>
						</td>
					</tr>
				{% endfor %}
				{% if not job.results %}
					<tr><td colspan=\"6\" class=\"muted\">No results yet.</td></tr>
				{% endif %}
			</tbody>
		</table>
	</div>

	<hr class=\"border-secondary\" />
	<h5 class=\"text-info\">Audit Trail (In-Memory)</h5>
	<pre class=\"explain mono border rounded p-3 bg-black mt-2\">{% for a in job.audit %}{{ a.ts }}  [{{ a.step }}]  item={{ a.item_index if a.item_index is not none else '-' }}  {{ a.message }}\n{% endfor %}{% if not job.audit %}—{% endif %}</pre>
</div>
</body>
</html>
"""


# -----------------------------
# Routes (UI)
# -----------------------------
@app.route("/", methods=["GET"])
def index() -> str:
		load_model()
		scenarios_map = _load_test_scenarios()
		selected_scenario = (request.args.get("scenario") or "").strip()
		payload = scenarios_map.get(selected_scenario) if selected_scenario else None

		default_payload = {
				"customer_no": "CUST_DEMO",
				"utilisation": 15.0,
				"dpd_days": 30,
				"cash_credit_ratio": 0.20,
				"cash_debit_ratio": 0.18,
				"inbound_cheque_bounce_count": 2,
				"inbound_cheque_bounce_amt": 5000,
				"outbound_cheque_bounce_count": 1,
				"outbound_cheque_bounce_amt": 2000,
				"total_amt_credit": 500000,
				"total_amt_debit": 450000,
				"no_of_banks": 3,
		}
		form = _coerce_payload(payload or default_payload)

		with JOBS_LOCK:
				jobs = list(JOBS.values())
		jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

		return render_template_string(
				TEMPLATE,
				kpis=_kpis_snapshot(),
				scenarios=sorted(list(scenarios_map.keys())),
				selected_scenario=selected_scenario,
				form=form,
				features=FEATURES,
				sync_result=None,
				submit_msg=None,
				jobs=jobs,
				accept_threshold=DECISION_THRESHOLDS["ACCEPT_MAX"],
				reject_threshold=DECISION_THRESHOLDS["REVIEW_MAX"],
		)


@app.route("/score/sync", methods=["POST"])
def score_sync() -> str:
		load_model()
		form = dict(request.form)
		payload: Dict[str, Any] = {"customer_no": (form.get("customer_no") or "").strip()}
		for f in FEATURES:
				payload[f] = form.get(f)

		job_id = "sync-" + uuid.uuid4().hex
		audit_sink: List[Dict[str, Any]] = []
		try:
				result = run_agentic_scoring(payload, job_id=job_id, item_index=0, audit_sink=audit_sink)
		except Exception as exc:
				result = {
						"customer_id": str(payload.get("customer_no") or "sync"),
						"decision": "ERROR",
						"policy": "exception",
						"risk_score": {"probability": 0.0, "probability_percentage": "0.00%", "bucket": "No Risk"},
						"model": "embedded",
						"explanation": f"Failed to score payload: {exc}",
				}

		scenarios_map = _load_test_scenarios()
		with JOBS_LOCK:
				jobs = list(JOBS.values())
		jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

		return render_template_string(
				TEMPLATE,
				kpis=_kpis_snapshot(),
				scenarios=sorted(list(scenarios_map.keys())),
				selected_scenario="",
				form=_coerce_payload(payload),
				features=FEATURES,
				sync_result=result,
				submit_msg=None,
				jobs=jobs,
				accept_threshold=DECISION_THRESHOLDS["ACCEPT_MAX"],
				reject_threshold=DECISION_THRESHOLDS["REVIEW_MAX"],
		)


@app.route("/jobs/submit", methods=["POST"])
def submit_job() -> Response:
		load_model()
		f = request.files.get("batch_file")
		if not f:
				return redirect(url_for("index"))

		try:
				items = _parse_batch_json(f.read())
				if not items:
						raise ValueError("No payloads found in uploaded JSON")

				job = _new_job(items)
				_append_audit(job, "job_queue", "Job queued")
				with JOBS_LOCK:
						JOBS[job["job_id"]] = job
						# Keep memory bounded.
						if len(JOBS) > MAX_RECENT_JOBS:
								oldest = sorted(JOBS.values(), key=lambda j: j.get("created_at") or "")[: max(0, len(JOBS) - MAX_RECENT_JOBS)]
								for j in oldest:
										JOBS.pop(j.get("job_id"), None)

				EXECUTOR.submit(_process_job, job["job_id"])
				msg = f"Submitted job {job['job_id']} with {job['total_items']} item(s)."

		except Exception as exc:
				msg = f"Batch upload failed: {exc}"

		scenarios_map = _load_test_scenarios()
		with JOBS_LOCK:
				jobs = list(JOBS.values())
		jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

		return Response(
				render_template_string(
						TEMPLATE,
						kpis=_kpis_snapshot(),
						scenarios=sorted(list(scenarios_map.keys())),
						selected_scenario="",
						form=_coerce_payload({}),
						features=FEATURES,
						sync_result=None,
						submit_msg=msg,
						jobs=jobs,
						accept_threshold=DECISION_THRESHOLDS["ACCEPT_MAX"],
						reject_threshold=DECISION_THRESHOLDS["REVIEW_MAX"],
				),
				mimetype="text/html",
		)


@app.route("/jobs/<job_id>", methods=["GET"])
def job_view(job_id: str) -> Response:
		with JOBS_LOCK:
				job = JOBS.get(job_id)
		if not job:
				return Response("Job not found", status=404)
		return Response(render_template_string(JOB_TEMPLATE, job=job), mimetype="text/html")


# -----------------------------
# Routes (API)
# -----------------------------
@app.route("/health", methods=["GET"])
def health() -> Response:
		load_model()
		return jsonify({"status": "healthy", "model_loaded": MODEL is not None, "model_path": str(MODEL_PATH)})


@app.route("/api/jobs", methods=["GET"])
def api_jobs() -> Response:
		with JOBS_LOCK:
				jobs = list(JOBS.values())
		out = []
		for j in jobs:
				out.append(
						{
								"job_id": j.get("job_id"),
								"status": j.get("status"),
								"created_at": j.get("created_at"),
								"started_at": j.get("started_at"),
								"finished_at": j.get("finished_at"),
								"total_items": j.get("total_items"),
								"completed_items": j.get("completed_items"),
								"accepted": j.get("accepted"),
								"rejected": j.get("rejected"),
								"avg_score": j.get("avg_score"),
								"max_score": j.get("max_score"),
								"error": j.get("error"),
						}
				)
		return jsonify(out)


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job(job_id: str) -> Response:
		with JOBS_LOCK:
				job = JOBS.get(job_id)
		if not job:
				return jsonify({"error": "job_not_found"}), 404
		return jsonify({k: v for k, v in job.items() if k not in {"items"}})


@app.route("/api/jobs/<job_id>/audit", methods=["GET"])
def api_job_audit(job_id: str) -> Response:
		with JOBS_LOCK:
				job = JOBS.get(job_id)
		if not job:
				return jsonify({"error": "job_not_found"}), 404
		return jsonify(job.get("audit") or [])


if __name__ == "__main__":
		debug_mode = os.getenv("TMU_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
		load_model()
		app.run(host="0.0.0.0", port=5065, debug=debug_mode, use_reloader=debug_mode)

