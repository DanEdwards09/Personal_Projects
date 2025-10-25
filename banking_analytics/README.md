# Grad Assessment Project: Card & Retail Banking Portfolio Review (90 Days)

**Scenario (Financial Services):**  
You’ve joined a retail bank’s Data & Analytics team. Senior management wants a short **90‑day performance review** of the card + current account portfolio that answers:
1) **Are we making more money from loyal customers?**  
2) **Which regions and segments are most valuable?**  
3) **Where are the key **risk signals** (chargebacks/churn), and what should we do about them in Q4?**

You are given realistic but synthetic datasets in `CSV` plus a SQL schema. Use **SQL for slicing/joins** and **Python for analysis + charts**. Your output should be a **5–8 slide deck** and a short **exec summary** (200–300 words).

---

## Datasets
- `customers.csv` – id, join_date, region, age, annual_income, segment, tenure_days  
- `accounts.csv` – account_id, customer_id, account_type, open_date  
- `transactions.csv` – transaction_id, account_id, txn_date, category, channel, amount (+/-)  
- `chargebacks.csv` – chargeback_id, account_id, transaction_id, cb_date, cb_amount, reason  
- `monthly_balances.csv` – derived ending balances per account per month  
- `customer_churn.csv` – last_txn_date, churned flag (no activity in last 90 days)

> Dates run through **2025-10-10** so you can use “last 30/60/90 days” windows.

---

## Core Tasks (what assessors look for)

### A) KPIs & Definitions (SQL first)
1. **Total revenue** last 90 days and by region/segment. (Use net inflows on transactions as a proxy; call out assumptions.)  
2. **Active customers** = at least 1 txn in last 30 days; **Loyal** = tenure ≥ 365 days.  
3. **Top 10% customers by value** (Pareto) and their region/segment mix.  
4. **Chargeback rate** = chargebacks / card transactions; trend last 3 months.  
5. **Balances**: avg monthly ending balance by account type and region.

### B) Exploration & Insight (Python)
6. Chart **revenue by loyalty & region** (stacked or grouped bars).  
7. **Cohort**: customers by join month; retention (still active in each month).  
8. **Churn analysis**: churn rate by age band and tenure buckets.  
9. **Risk signals**: which categories/channels are associated with chargebacks?  
10. **Unit economics**: estimate gross margin per customer (state assumptions).

### C) Recommendation & Communication
11. Draft a **one‑page memo** summarising findings and **3 actions** (e.g., fee waivers for high‑value cohorts; fraud rules for specific channels; win‑back for at‑risk tenure band).  
12. Build a **compact dashboard** (optional) with 3–4 charts for a live demo.

---

## Deliverables
- `notebooks/portfolio_review.ipynb` – analysis with SQL-style joins in pandas  
- `slides/portfolio_review.pptx` or a short markdown report  
- `outputs/` charts (`.png`) used in slides  
- `sql/answers.sql` – your key queries

---

## Acceptance Criteria (rubric)
- **Clarity (25%)**: clean queries, tidy charts, labelled axes, stated assumptions  
- **Rigor (35%)**: correct windowing (30/60/90d), sensible revenue proxy, cohort logic, sanity checks  
- **Business (25%)**: prioritised recommendations tied to quantified impact  
- **Communication (15%)**: 3–4 sentence exec summary, one slide per key message

---

## Starter Questions (map to assessment skills)
- SQL: join customers→accounts→transactions; window functions for top 10%; group by region/segment; 90‑day filters.  
- Python: cohort retention, churn by buckets, chargeback correlations, Pareto charts.  
- Visuals: revenue by region×loyalty; churn by tenure; chargeback rate trend.  
- Memo: “What would you do next quarter and why?”

---

## Getting Started
- Use `schema.sql` if you want to load to SQLite/Postgres.  
- Or read CSVs directly in Python: see `notebooks/portfolio_review.ipynb` skeleton.

Good luck — this mirrors the **coding + analysis + communication** mix you’ll face in assessment centres.