# Component 1: Next Best Action System – Technical Solution
**Candidate:** Md Marghub Akhtar | **Role:** Senior Data Scientist 

---

## 1. Problem Framing & Exploration

**Strategic Reframe:** From *"Who will churn?"* to *"Which action creates maximum incremental value per customer?"*

**Exploratory Focus:**
- Churn risk distribution by tenure, industry, engagement → Identify economically "worth-saving" customers (telco/subscription industries: 70-80% revenue from top 20-30%)
- Historical intervention effectiveness and costs → Baseline: 15-35% response (automated), 50-70% (personal outreach)
- **Critical insight:** Does upselling low-risk customers increase churn? (Research shows aggressive upselling can increase churn 5-10% if engagement is low)

**Key Discovery:** Framework must balance retention, growth, and efficiency without conflicts;not all high-risk customers warrant intervention.

---

## 2. Analytical & Modeling Approach

### 2.1   Hybrid Framework: Analytics-First, Selective Modeling

**3D Customer Segmentation:**
- **Risk:** High (>60%), Medium (30-60%), Low (<30%) *[Subscription business tertiles]*
- **CLV:** Annual Revenue × (1 - Churn Prob) × 3 yrs *[Gartner B2B benchmark: 2.5-4 yrs. Also, add contract length, service mix, renewal fees]*
- **Engagement:** Login frequency + feature adoption + support interaction *[>75 = top quartile; ≤75 = dissatisfaction signal]*

**Segment-Action Map:**
| Segment | Risk | Value | Engagement | Action | Rationale |
|---------|------|-------|------------|--------|-----------|
| Critical Save | High | High ($50K+ CLV) | Any | Account Manager Call + Offer | High ROI, personal touch |
| Cost-Efficient Retention | High | Low (<$10K CLV) | Any | Automated Discount Email | Cost-effective at scale |
| Upsell Opportunity | Low | High | High (>75) | Targeted Upsell | Engaged, receptive, safe |
| Monitor Only | Low | High | Low (≤75) | No Action | Annoyance risk > opportunity |
| Standard Service | Any | Low | Any | No Action | Resource priority |

### 2.2   Expected Value Decision Framework

For each customer-action pair: **EV = P(Success) × Incremental_Value - Cost_of_Action**

**Action Assignment:** Rank all customers by EV → Allocate to top N within constraints:
- Budget limit: $100K/month for retention discounts
- Capacity: 50 account manager calls/week
- Policy: Max 2 contacts per customer per quarter

### 2.3   Selective Predictive Models

**Model 1 – Upsell Propensity (XGBoost):** P(Upsell) using engagement, feature requests, tenure, industry  
**Model 2 – Treatment Response (Multi-class):** Optimal channel (call/email/offer) per customer

**Rationale:** Considered logistic regression (simpler) and deep learning (complex). Chose XGBoost: balances performance, interpretability, proven at Capital One/Uber. **Not uplift initially:** Needs experimental data (unavailable; only 20-30% of companies have this). **Phased:** Propensity now → A/B tests → Uplift in 6-9 months.

**Retention-Upsell Conflict Resolution:**
```python
IF churn_risk >= 60%: action = "Retention"  # Priority always
ELIF churn_risk < 20% AND upsell_propensity > 0.4 AND engagement > 75: action = "Upsell"
ELIF churn_risk < 30% AND engagement <= 75: action = "Monitor"  # Don't risk annoyance
```

---

## 3. Evaluation & Experimentation

### 3.1 Pre-Launch Validation
- **Historical simulation:** Apply NBA to past 6 months → "What if we had used this?" → Estimate incremental revenue
- **Model metrics:** Precision@100 (upsell), Expected Value of top 500 recommendations (business-aligned)

### 3.2 A/B Test Design (Critical for Proof)

| Aspect | Details |
|--------|---------|
| **Groups** | Control (30%): Business as usual \| Treatment (70%): NBA recommendations |
| **Stratification** | By value tier (High/Med/Low) × risk tier (High/Med/Low) = 9 strata |
| **Duration** | 12 weeks (covers renewal cycles, seasonal effects) |
| **Primary Metrics** | • Retention rate delta<br>• Incremental revenue per customer<br>• Cost per retention/upsell |
| **Secondary Metrics** | • NPS/CSAT change<br>• Sales team efficiency |
| **Statistical Rigor** | Power = 90% to detect 5% lift, α=0.05, Bayesian early stopping |

**Success Criteria (Benchmarked):**
- **Minimum Viable Success:** +3% retention (conservative; achievable with basic segmentation), +10% cost efficiency, neutral NPS
- **Target:** +8-10% retention (*Industry benchmarks: Etisalat 8%, Vodafone 12%, McKinsey telco/subscription avg 7-15%*), +$3-5M annual revenue (*assumption: 5,000 renewal customers @ $15K avg CLV*), 30% fewer wasted interventions (*baseline: eliminate offers to "sure things" segment*)

---

## 4. Operationalization & Learning

**Production Architecture (Daily Batch):**
1. **Data Pipeline:** Feature store (Feast/Tecton) ingests customer attributes, engagement signals, churn scores → Data quality checks (schema validation, null detection) → Feature engineering
2. **Scoring Engine:** Containerized NBA models (Docker) orchestrated on Kubernetes/Azure ML → Batch inference on renewal window customers (30-90 days out)
3. **Integration:** Recommendations pushed to CRM via REST API with EV ranking, priority flags, action rationale
4. **Consumption:** Sales/retention teams access prioritized queues through CRM dashboards

**Deployment Strategy:**
- **CI/CD:** GitHub Actions for automated testing → Model registry (MLflow) for versioning → Canary deployment (10% → 50% → 100% traffic)
- **Real-time API (Phase 2):** FastAPI endpoint for on-demand scoring, <200ms p95 latency, auto-scaling
- **Rollback:** Blue-green deployment with instant fallback to previous model version if issues detected

**Monitoring & Observability:**
- **Model Performance:** Weekly validation on holdout set; drift detection (PSI for features, KL divergence for predictions); auto-alerts at >10% accuracy drop
- **Business Metrics:** Daily dashboard tracking retention rate, recommendation acceptance, revenue impact, cost per action
- **System Health:** Inference latency, throughput, error rates; Prometheus metrics + Grafana dashboards

**Feedback Loop & Continuous Improvement:**
- Log all recommendations + outcomes (accepted/rejected/churned/renewed) → Store in feature store
- Retrain upsell propensity (monthly), treatment response (bi-weekly) using fresh data
- Quarterly segment review: Adjust thresholds, add new features based on patterns

**Governance & Compliance:**
- Data: PII encrypted at rest/transit, GDPR-aligned retention, audit trails for all decisions
- UAE compliance: Decision transparency, explainable action rationale (SHAP values available)
- Human override: Account managers can override with justification (logged for model improvement)

**Risk Mitigation:**
- *A/B test underperforms:* Segment-level analysis, threshold refinement, iterate before abandoning
- *Model degrades:* Auto-alerts, weekly holdout validation, immediate investigation protocol
- *Data quality issues:* 10% pilot validation, schema checks, anomaly detection before scaling
- *Priority shifts:* Modular EV formula allows re-weighting objectives (e.g., growth vs retention)

---

## Summary: Key Decisions & Impact

**What Leadership Decides Differently on Day 1:** Instead of reactive, ad-hoc outreach, leadership now allocates resources systematically—prioritizing by expected value, respecting capacity limits, and measuring incremental impact through controlled experiments. Every dollar spent is justified by customer economics, not intuition.

**Design Choices:**
1. Analytics-first segmentation (interpretable, works with available data)
2. Expected value + constraints (budget, capacity) → Realistic deployment
3. Phased sophistication: Propensity now → Collect A/B data → Uplift in 6 months
4. Conflict resolution: Retention always wins over upsell when risk ≥60%

**Timeline:** Weeks 1-3 (Explore) → 4-6 (Build) → 7-8 (Pilot) → 9-20 (A/B Test) → 6+ months (Uplift upgrade)

**Expected Impact:** +8-12% retention | $4-7M annual revenue | 30-40% cost efficiency | Neutral-positive NPS

**Data Assumptions:** Customer attributes (tenure, industry), engagement signals (logins, support), historical actions/responses, current churn scores (from existing model). No experimental data currently → Will collect via A/B test.

---


