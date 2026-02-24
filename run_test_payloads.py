from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RunResult:
    name: str
    mode: str
    http_status: Optional[int]
    ok: bool
    output: Dict[str, Any]
    audit: List[Dict[str, Any]]


def fmt_audit(audit: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for event in audit:
        timestamp = event.get("timestamp", "-")
        step = event.get("step", "-")
        item = event.get("item", "-")
        message = event.get("message", "")
        lines.append(f"{timestamp}  [{step}]  item={item}  {message}")
    return "\n".join(lines)


def is_batch_payload(payload: Any, filename: str) -> bool:
    if filename.startswith("batch_"):
        return True
    if isinstance(payload, list):
        return True
    if isinstance(payload, dict) and isinstance(payload.get("customers"), list):
        return True
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return True
    return False


def http_json(method: str, url: str, body: Any | None) -> Tuple[int, Dict[str, Any]]:
    """Stdlib-only JSON HTTP helper (no requests dependency)."""
    import urllib.error
    import urllib.request

    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        raw = exc.read().decode("utf-8") if exc.fp else ""
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed: {exc}") from exc

    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"raw": raw}

    return status, payload


def run_local_single(payload: Dict[str, Any]) -> RunResult:
    from credit_scoring_agent import add_audit_event, load_model, now_iso, run_agentic_item

    load_model()

    audit: List[Dict[str, Any]] = []
    add_audit_event(audit, "flow", 0, "Starting single-item flow")

    item = payload["customer"] if isinstance(payload.get("customer"), dict) else payload
    result = run_agentic_item(item, 0, audit)

    add_audit_event(audit, "flow", 0, "Completed single-item flow")

    ok = result.get("status") != "ERROR"
    output = {
        "timestamp": now_iso(),
        "result": result,
    }
    return RunResult(
        name=str(payload.get("customer_no") or result.get("customer_id") or "single"),
        mode="local-single",
        http_status=None,
        ok=ok,
        output=output,
        audit=audit,
    )


def run_local_batch(payload: Any) -> RunResult:
    from credit_scoring_agent import add_audit_event, load_model, now_iso, parse_batch_payload, run_agentic_item, summarize_results

    load_model()

    items = parse_batch_payload(payload)

    audit: List[Dict[str, Any]] = []
    add_audit_event(audit, "flow", None, "Batch flow started")

    results: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        item_audit: List[Dict[str, Any]] = []
        add_audit_event(item_audit, "flow", index, "Starting item flow")
        result = run_agentic_item(item, index, item_audit)
        add_audit_event(item_audit, "flow", index, "Completed item flow")
        audit.extend(item_audit)
        results.append(result)

    add_audit_event(audit, "flow", None, "Batch flow completed")

    summary = summarize_results(results)
    ok = summary.get("errors", 0) == 0

    output = {
        "timestamp": now_iso(),
        "summary": summary,
        "results": results,
    }

    return RunResult(
        name="batch",
        mode="local-batch",
        http_status=None,
        ok=ok,
        output=output,
        audit=audit,
    )


def run_http_single(base_url: str, payload: Dict[str, Any]) -> RunResult:
    status, data = http_json("POST", f"{base_url.rstrip('/')}/score", payload)
    audit = data.get("audit") if isinstance(data, dict) else None
    return RunResult(
        name=str(payload.get("customer_no") or "single"),
        mode="http-single",
        http_status=status,
        ok=200 <= status < 300,
        output=data if isinstance(data, dict) else {"data": data},
        audit=audit if isinstance(audit, list) else [],
    )


def run_http_batch(base_url: str, payload: Any) -> RunResult:
    status, data = http_json("POST", f"{base_url.rstrip('/')}/score/batch", payload)
    audit = data.get("audit") if isinstance(data, dict) else None
    return RunResult(
        name="batch",
        mode="http-batch",
        http_status=status,
        ok=200 <= status < 300,
        output=data if isinstance(data, dict) else {"data": data},
        audit=audit if isinstance(audit, list) else [],
    )


def print_result(file_name: str, result: RunResult, show_full_output: bool, show_audit: bool) -> None:
    prefix = "OK" if result.ok else "ERROR"
    http_part = f" http={result.http_status}" if result.http_status is not None else ""
    print(f"\n{'=' * 72}")
    print(f"{file_name}  [{prefix}]{http_part}  ({result.mode})")
    print(f"{'=' * 72}")

    output = result.output

    # Try to pull key decision bits for human-readable summary
    if isinstance(output, dict) and "result" in output and isinstance(output["result"], dict):
        r = output["result"]
        risk = r.get("risk_score") or {}
        print(f"customer_id: {r.get('customer_id')}")
        print(f"probability: {risk.get('probability')} ({risk.get('probability_percentage')})")
        print(f"bucket: {risk.get('bucket')}")
        print(f"decision: {r.get('decision')}")
        print(f"policy: {r.get('policy')}")
        print(f"recommendation: {r.get('recommendation')}")
    elif isinstance(output, dict) and "results" in output:
        summary = output.get("summary")
        if isinstance(summary, dict):
            print(
                "summary: "
                + ", ".join(
                    f"{k}={summary.get(k)}"
                    for k in ["total", "accepted", "conditional_accept", "review", "rejected", "errors", "avg_score", "max_score"]
                    if k in summary
                )
            )
        results = output.get("results")
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                if "customer_id" in row and "risk_score" in row and "decision" in row and isinstance(row["decision"], dict):
                    print(
                        f"- {row.get('customer_id')}: "
                        f"p={(row.get('risk_score') or {}).get('probability')} "
                        f"bucket={(row.get('risk_score') or {}).get('bucket')} "
                        f"decision={(row.get('decision') or {}).get('status')}"
                    )
                else:
                    risk = row.get("risk_score") or {}
                    print(
                        f"- {row.get('customer_id')}: "
                        f"p={risk.get('probability')} "
                        f"bucket={risk.get('bucket')} "
                        f"decision={row.get('decision')}"
                    )

    if show_full_output:
        print("\n--- output ---")
        print(json.dumps(output, indent=2))

    if show_audit:
        print("\n--- audit trail ---")
        print(fmt_audit(result.audit) if result.audit else "(no audit events)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all test_payloads JSON fixtures and print status, decisions, and audit trail.")
    parser.add_argument("--dir", default="test_payloads", help="Directory containing JSON payloads")
    parser.add_argument(
        "--url",
        default=None,
        help="If set, run via HTTP against this base URL (example: http://localhost:5063). Otherwise runs locally in-process.",
    )
    parser.add_argument("--full", action="store_true", help="Print full JSON output for each payload")
    parser.add_argument("--no-audit", action="store_true", help="Do not print audit trail")

    args = parser.parse_args()

    payload_dir = Path(args.dir)
    if not payload_dir.exists() or not payload_dir.is_dir():
        print(f"Payload dir not found: {payload_dir}", file=sys.stderr)
        return 2

    json_files = sorted(payload_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {payload_dir}")
        return 2

    total = 0
    failures = 0

    for path in json_files:
        total += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures += 1
            print(f"\n{path.name}: ERROR reading JSON: {exc}")
            continue

        batch = is_batch_payload(payload, path.name)

        try:
            if args.url:
                result = run_http_batch(args.url, payload) if batch else run_http_single(args.url, payload)
            else:
                result = run_local_batch(payload) if batch else run_local_single(payload)
        except Exception as exc:
            failures += 1
            print(f"\n{path.name}: ERROR running payload: {exc}")
            continue

        if not result.ok:
            failures += 1

        # Wrap local single result to match print summary
        if result.mode.startswith("local") and not batch:
            print_result(path.name, result, show_full_output=args.full, show_audit=not args.no_audit)
        else:
            print_result(path.name, result, show_full_output=args.full, show_audit=not args.no_audit)

        # Small pause to keep output readable for HTTP mode
        if args.url:
            time.sleep(0.05)

    print(f"\nDone. total={total} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
