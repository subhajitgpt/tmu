# n8n Cloud – Credit Scoring Flow (UI Build Sheet)

This is a UI-deliverable blueprint for implementing your credit-scoring orchestration in **n8n Cloud**, using Python only for model inference.

It is based on the workflow already provided in this repo: `n8n/credit_scoring_workflow.json` and the Python inference service `n8n/ml_service.py`.

---

## 0) Key Cloud Constraint (Important)

In **n8n Cloud**, the workflow runner does **not** have network access to your laptop’s `http://localhost:5064`.

You must do one of the following so the **HTTP Request → ML Service** node can reach Python:

- **Production-style (recommended):** Deploy `ml_service.py` somewhere reachable over HTTPS (Render/Fly.io/Railway/Cloud Run/etc.)
- **Quick test:** Use a tunnel from your local machine (ngrok / Cloudflare Tunnel) and point n8n to the tunnel URL.

Target endpoint needed by n8n:
- `GET <ML_BASE_URL>/health`
- `POST <ML_BASE_URL>/predict`

### 0.1) If Python is running locally: expose it via a tunnel

You have two common options.

#### Option A — Cloudflare Tunnel (quick, no account for temporary URL)
1. Install `cloudflared`.
2. Run:
  - `cloudflared tunnel --url http://localhost:5064`
3. Copy the generated `https://<random>.trycloudflare.com` URL.
4. In n8n Cloud, set the ML URL to:
  - `https://<random>.trycloudflare.com/predict`

#### Option B — ngrok (quick, requires ngrok installed)
1. Run:
  - `ngrok http 5064`
2. Copy the `https://<random>.ngrok-free.app` URL.
3. In n8n Cloud, set the ML URL to:
  - `https://<random>.ngrok-free.app/predict`

### 0.2) Secure the tunneled endpoint (recommended)

Because a tunnel creates a public URL to your local machine, enable a simple API key on the ML service.

This repo’s `n8n/ml_service.py` supports this:
- Set an environment variable on the machine running Python: `ML_API_KEY`.
- n8n must send header: `X-ML-API-Key: <your key>`.

---

## 1) Workflow Summary

**Workflow name:** `Credit Scoring Workflow (Cloud)`

**Trigger:** Webhook (POST)

**Input payload (from caller):**
```json
{
  "customer_no": "CUST001",
  "utilisation": 15.5,
  "dpd_days": 5,
  "cash_credit_ratio": 0.18,
  "cash_debit_ratio": 0.15,
  "inbound_cheque_bounce_count": 0,
  "inbound_cheque_bounce_amt": 0,
  "outbound_cheque_bounce_count": 0,
  "outbound_cheque_bounce_amt": 0,
  "total_amt_credit": 750000,
  "total_amt_debit": 680000,
  "no_of_banks": 2
}
```

**Decision thresholds (as implemented):**
- `REJECT` if `probability >= 0.80`
- `REVIEW` if `0.70 <= probability < 0.80`
- `CONDITIONAL_ACCEPT` if `0.50 <= probability < 0.70`
- `ACCEPT` if `probability < 0.50`

---

## 2) Fastest Path: Import Then Edit (Recommended)

1. In n8n Cloud, create a new workflow.
2. Import `n8n/credit_scoring_workflow.json` (upload file / paste JSON).
3. Update these nodes:

### A) Node: `ML Service - Get Prediction`
- **URL:** change from `http://localhost:5064/predict` → `https://<your-ml-service-domain>/predict`
- **Method:** `POST`
- **Body Content Type:** JSON
- **Body:** keep as-is (it sends the output of `Validate Input`, which matches `ml_service.py` contract)

### B) Node: `Log to PostgreSQL` (optional)
This node is **disabled** in the JSON. If you leave it disabled, the workflow path that connects through it may not continue as expected.

Choose one:
- **Option 1 (simplest):** Enable it and configure credentials.
- **Option 2:** Keep it disabled and rewire:
  - Connect `Merge All Decisions` → `IF High Risk?` directly.

### C) Node: `Email Compliance Team` and `Slack Credit Officers` (optional)
These are also disabled in the JSON.

- Enable + configure credentials if you want alerts.
- If left disabled, connect the `IF High Risk?` TRUE branch directly to `Respond to Webhook`.

### D) Webhook authentication (recommended in Cloud)
Open the `Webhook - Credit Application` node:
- Turn on authentication (pick one)
  - **Header Auth**: require `X-API-Key: <secret>`
  - or **Basic Auth**

---

## 3) Node-by-Node “UI Build” Specification

Use this section if you prefer building from scratch (or validating the imported workflow).

### Node 1 — `Webhook - Credit Application`
- **Type:** Webhook
- **HTTP Method:** `POST`
- **Path:** `credit-score`
- **Response Mode:** `Using "Respond to Webhook" node`
- **Auth:** Header Auth (recommended)

### Node 2 — `Validate Input`
- **Type:** Code (JavaScript) *(or “Function” if your n8n still has it)*
- **Purpose:** Validate required fields + types + ranges; map input → `customer_id` + `features`.
- **Input location:** `items[0].json.body`
- **Output JSON:**
  - `customer_id`
  - `features` (11 numeric fields)

*(Tip: You can copy the code from the imported workflow node directly.)*

### Node 3 — `ML Service - Get Prediction`
- **Type:** HTTP Request
- **Method:** `POST`
- **URL:** `https://<ML_BASE_URL>/predict`
- **Send Body:** true
- **Body:** JSON (send the entire current `$json`)

### Node 4 — `Calculate Risk Bucket`
- **Type:** Code (JavaScript)
- **Purpose:** Compute bucket + decision status and attach timestamps.
- **Output JSON:**
  - `customer_id`, `probability`, `probability_percentage`, `bucket`, `status`, `shap_values`, `features`, `timestamp`

### Node 5 — `Route By Decision`
- **Type:** Switch
- **Rules:**
  - `probability >= 0.8` → output `reject`
  - `probability >= 0.7` → output `review`
  - `probability >= 0.5` → output `conditional`
  - `probability < 0.5` → output `accept`

### Nodes 6–9 — Set per decision
- **Type:** Set
- **Nodes:**
  - `Set REJECT Response`
  - `Set REVIEW Response`
  - `Set CONDITIONAL Response`
  - `Set ACCEPT Response`
- **Purpose:** Fill `decision.*` fields (`status`, `recommendation`, `risk_level`, `action`, `explanation`).

### Node 10 — `Merge All Decisions`
- **Type:** Merge
- **Mode:** combine

### Node 11 — `Log to PostgreSQL` (optional)
- **Type:** Postgres
- **Operation:** insert
- **Table:** `credit_decisions`
- **Columns:** `customer_id, probability, bucket, decision_status, timestamp`

### Node 12 — `IF High Risk?`
- **Type:** IF
- **Condition:** `probability >= 0.7`

### Node 13 — `Email Compliance Team` (optional)
- **Type:** Email
- **To:** compliance distribution list
- **Body:** include customer + probability + decision

### Node 14 — `Slack Credit Officers` (optional)
- **Type:** Slack
- **Channel:** credit-risk / underwriting
- **Text:** include customer + probability + bucket + explanation

### Node 15 — `Respond to Webhook`
- **Type:** Respond to Webhook
- **Respond With:** JSON
- **Response body:** structured response (customer_id, timestamp, risk_score, decision, model_version)

---

## 4) Suggested Production Hardening (Keep Minimal)

- Add webhook authentication (Header Auth) and rotate keys.
- Add an allowlist of caller IPs if your architecture permits.
- Set HTTP Request timeout (already set to 10s in the JSON).
- Ensure your ML service is HTTPS and protected (private network / auth token / mTLS if needed).

---

## 5) Test in Cloud

1. Activate the workflow.
2. Copy the **Production webhook URL** from the Webhook node.
3. POST one of the payloads from `test_payloads/`.

---

## 6) What I Need From You (to finalize exact UI fields)

Reply with:
1) Where will the Python ML service live? (public URL or tunnel URL)
2) Which notifications do you want enabled? (Email / Slack / both / neither)
3) Do you want database logging? If yes, Postgres connection details (host/db) and table name.
