# Data Collection for NBA System: FAANG Standards

##  Standard Assumptions (Based on Meta, Google, Amazon, Uber)

### Infrastructure That Already Exists:
```
 Data Warehouse (Snowflake/BigQuery/Redshift)
 Streaming Pipeline (Kafka/Kinesis)
 Feature Store (Tecton/Feast or in-house)
 ML Platform (Sagemaker/Vertex AI/in-house)
 Experimentation Platform (A/B testing framework)
 Real-time logging infrastructure
 CRM/Business systems with APIs
```

---

##  What Data to Collect (Detailed Specification)

### 1. Customer Profile Data (Batch - Daily)

**Source:** CRM, Billing Systems, Contract Management

| Field | Type | Example | Frequency | Storage |
|-------|------|---------|-----------|---------|
| `customer_id` | string | "CUST_12345" | Daily | Dimension table |
| `tenure_months` | int | 24 | Daily | Derived |
| `industry` | categorical | "Finance" | On change | Slowly changing dimension |
| `contract_value_annual` | decimal | 75000.00 | Daily | Fact table |
| `contract_type` | categorical | "Enterprise" | On change | Dimension |
| `renewal_date` | date | "2026-06-30" | On change | Critical! |
| `account_owner` | string | "sales_rep_id_42" | On change | For routing |
| `company_size` | int | 250 | Quarterly | Firmographic |
| `services_subscribed` | array | ["Service_A", "Service_B"] | On change | JSON field |

**Collection Method:**
```python
# Daily batch ETL
SELECT 
    customer_id,
    DATEDIFF(month, signup_date, CURRENT_DATE) as tenure_months,
    industry,
    annual_contract_value,
    ...
FROM crm.customers
WHERE status = 'active'
```

---

### 2. Engagement Signals (Streaming - Real-time)

**Source:** Product Analytics, Application Logs, Support Systems

| Signal | Type | Measurement | Window | Weight |
|--------|------|-------------|--------|--------|
| Login frequency | Count | Logins per week | 30 days | High |
| Feature adoption | Ratio | Features used / Total features | 90 days | High |
| Session duration | Minutes | Avg session length | 30 days | Medium |
| API calls | Count | API usage | 7 days | High |
| Support tickets | Count | Tickets opened | 90 days | Negative |
| Support sentiment | Score | NPS/CSAT | Last interaction | High |
| Feature requests | Count | Requests submitted | 180 days | Positive |
| Documentation views | Count | Help pages visited | 30 days | Medium |
| User invites sent | Count | New users added | 90 days | Very High |

**Engagement Score Calculation:**
```python
engagement_score = (
    0.25 * normalize(login_frequency) +
    0.30 * feature_adoption_ratio +
    0.15 * normalize(session_duration) +
    0.10 * normalize(api_calls) +
    0.10 * (1 - normalize(support_tickets)) +  # Inverted
    0.05 * normalize(documentation_views) +
    0.05 * normalize(user_invites)
) * 100  # Scale to 0-100
```

**Collection Method (Real-time Stream):**
```python
# Event stream processing (Kafka/Kinesis)
{
    "event_type": "user_login",
    "timestamp": "2026-02-24T10:30:00Z",
    "customer_id": "CUST_12345",
    "user_id": "USER_789",
    "session_id": "SESSION_XYZ",
    "metadata": {
        "feature_accessed": "dashboard",
        "duration_minutes": 15
    }
}

# Aggregated hourly into feature store
```

---

### 3. Churn Probability (Batch - Weekly)

**Source:** Existing ML Model (Given in problem)

| Field | Type | Description | Update Frequency |
|-------|------|-------------|------------------|
| `churn_probability` | float (0-1) | Risk score | Weekly |
| `churn_model_version` | string | "v2.3.1" | On change |
| `churn_factors` | JSON | Top risk drivers | Weekly |
| `prediction_timestamp` | timestamp | When scored | Weekly |

**FAANG Standard:** Most companies rescore weekly (balance freshness vs compute cost)

---

### 4. Historical Actions & Outcomes (Event-driven)

**Source:** CRM, Marketing Automation, Sales Systems

**Action Log Table:**
```sql
CREATE TABLE customer_actions (
    action_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,  -- 'retention_call', 'retention_email', 'upsell_offer'
    action_channel VARCHAR,         -- 'phone', 'email', 'in_app'
    action_timestamp TIMESTAMP NOT NULL,
    action_cost DECIMAL(10,2),      -- $500 for call, $50 for email
    campaign_id VARCHAR,
    sales_rep_id VARCHAR,
    offer_details JSON,             -- Discount %, terms, etc.
    
    -- Outcome tracking
    response_timestamp TIMESTAMP,   -- When customer responded
    outcome VARCHAR,                -- 'accepted', 'rejected', 'no_response'
    outcome_timestamp TIMESTAMP,    -- When outcome determined
    revenue_impact DECIMAL(10,2),   -- Incremental revenue
    
    -- Metadata
    ab_test_group VARCHAR,          -- 'control', 'treatment' (if in experiment)
    experiment_id VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Collection at Different Stages:**

```python
# Stage 1: Action Taken (Immediate)
log_action({
    'customer_id': 'CUST_12345',
    'action_type': 'retention_call',
    'action_timestamp': datetime.now(),
    'action_cost': 500.00,
    'sales_rep_id': 'REP_42',
    'ab_test_group': 'treatment',
    'experiment_id': 'NBA_Q1_2026'
})

# Stage 2: Immediate Response (Minutes/Hours)
log_response({
    'action_id': 'ACTION_789',
    'response_timestamp': datetime.now(),
    'outcome': 'accepted',  # Customer agreed to stay
})

# Stage 3: Long-term Outcome (30/60/90 days)
log_outcome({
    'action_id': 'ACTION_789',
    'outcome_timestamp': datetime.now() + timedelta(days=90),
    'churned': False,
    'revenue_impact': 75000.00,  # Retained full contract
})
```

---

### 5. A/B Test Experimental Data (During Test Period)

**Source:** Experimentation Platform

**Experiment Assignment Table:**
```sql
CREATE TABLE experiment_assignments (
    experiment_id VARCHAR,
    customer_id VARCHAR,
    variant VARCHAR,           -- 'control', 'treatment_v1', etc.
    assignment_timestamp TIMESTAMP,
    eligibility_criteria JSON, -- Why they qualified
    stratification_key VARCHAR -- Segment for balanced split
);
```

**Treatment Application Table:**
```sql
CREATE TABLE treatment_applications (
    experiment_id VARCHAR,
    customer_id VARCHAR,
    recommended_action VARCHAR,  -- What NBA system said
    applied_action VARCHAR,      -- What actually happened (may differ)
    override_reason VARCHAR,     -- If human overrode system
    application_timestamp TIMESTAMP
);
```

**Outcome Measurement:**
- **Primary Metrics:**
  - Retention rate (binary: retained at 90 days)
  - Revenue per customer (continuous)
  - Cost per intervention (continuous)
  
- **Secondary Metrics:**
  - NPS change (survey)
  - Sales team efficiency (time per customer)
  - Customer complaints (count)

---

##  Data Pipeline Architecture (FAANG Standard)

### Real-time Stream Processing:
```
User Actions → Kafka → Flink/Spark Streaming → Feature Store
                                              ↓
                                        Engagement Scores
                                              ↓
                                        Ready for inference
```

### Batch Processing (Daily):
```
CRM/Billing DB → Airflow DAG → Data Warehouse → Feature Engineering
                                    ↓
                              Feature Store
                                    ↓
                            ML Model Serving
```

### Feature Store Schema:
```python
# Feast feature definitions
from feast import Entity, Feature, FeatureView, ValueType

customer = Entity(
    name="customer",
    value_type=ValueType.STRING,
    description="Customer ID"
)

customer_features = FeatureView(
    name="customer_profile",
    entities=["customer"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="tenure_months", dtype=ValueType.INT64),
        Feature(name="annual_revenue", dtype=ValueType.FLOAT),
        Feature(name="engagement_score", dtype=ValueType.FLOAT),
        Feature(name="churn_probability", dtype=ValueType.FLOAT),
        # ... more features
    ],
    online=True,  # Enable real-time serving
    batch_source=bigquery_source  # Offline training
)
```

---

##  Data Collection Timeline

### Week 1-2: Data Audit & Pipeline Setup
```
Tasks:
├─ Identify all data sources (CRM, product, support)
├─ Document schemas and access patterns
├─ Set up data quality monitoring
├─ Create feature store tables
└─ Establish baselines

Deliverables:
├─ Data catalog (what exists, where, how fresh)
├─ Data quality dashboard
└─ Sample datasets for EDA
```

### Week 3-8: Historical Analysis & Model Building
```
Data Used:
├─ Past 12 months of customer data
├─ Historical interventions (if any)
├─ All engagement metrics
└─ Churn model scores

Collection Method:
├─ One-time batch extraction
├─ Load into analytics environment
└─ No real-time needed yet

Output:
├─ 10,000+ customer records
├─ Segmentation rules defined
├─ Initial propensity models trained
└─ NBA logic implemented
```

### Week 9-20: A/B Test (Active Data Collection)
```
Data Collection Per Customer:
├─ Pre-test: All baseline features
├─ Assignment: Control/Treatment (logged)
├─ During: Actions taken, costs incurred
└─ Post-test: Outcomes at 30/60/90 days

Collection Infrastructure:
├─ Event logging for every action
├─ Daily sync with CRM for outcomes
├─ Weekly snapshot for analysis
└─ Real-time dashboard for monitoring

Volume:
├─ ~2,500 customers in test
├─ ~4 actions per customer (avg)
├─ = 10,000 action-outcome pairs
└─ Perfect for uplift modeling
```

### Week 21+: Production Data Collection
```
Continuous Logging:
├─ Every NBA recommendation → Logged
├─ Every action taken → Logged  
├─ Every outcome → Tracked
├─ Monthly model retraining
└─ Quarterly performance reviews

Data Retention:
├─ Raw events: 2 years
├─ Aggregated features: 5 years
├─ Model predictions: 1 year
└─ Compliance: Per GDPR/UAE law
```

---

##  Data Volume Estimates (FAANG Scale)

### Example: 50,000 Active Customers

| Data Type | Daily Volume | Monthly Volume | Storage |
|-----------|--------------|----------------|---------|
| Customer profiles | 50K rows | 1.5M rows | 500 MB |
| Engagement events | 2M events | 60M events | 15 GB |
| Churn scores | 50K predictions | 200K predictions | 10 MB |
| NBA recommendations | 5K/day | 150K/month | 100 MB |
| Action outcomes | 1K/day | 30K/month | 50 MB |
| **Total** | **~80 MB/day** | **~2.4 GB/month** | **30 GB/year** |

**Compute Costs (AWS):**
- Data storage: $30/month (S3)
- Streaming pipeline: $500/month (Kinesis/Flink)
- Feature store: $200/month (Redis/DynamoDB)
- Model serving: $300/month (Lambda/SageMaker)
- **Total: ~$1,000/month** (very reasonable at FAANG scale)

---

##  Data Quality Standards

### FAANG-Level Validation:

```python
class DataQualityChecks:
    """Applied at ingestion and daily batch"""
    
    def validate_customer_profile(self, df):
        assert df['customer_id'].is_unique, "Duplicate customer IDs"
        assert df['customer_id'].notnull().all(), "Missing customer IDs"
        assert (df['tenure_months'] >= 0).all(), "Negative tenure"
        assert (df['annual_revenue'] >= 0).all(), "Negative revenue"
        assert df['churn_probability'].between(0, 1).all(), "Invalid churn prob"
        
    def validate_engagement_score(self, df):
        assert df['engagement_score'].between(0, 100).all(), "Score out of range"
        assert df['engagement_score'].notnull().sum() / len(df) > 0.95, "Too many nulls"
        
    def validate_action_log(self, df):
        assert df['action_type'].isin(['retention_call', 'retention_email', 
                                       'upsell_offer', 'none']).all(), "Invalid action"
        assert df['action_cost'].notnull().all(), "Missing cost"
        assert df['action_timestamp'] <= datetime.now(), "Future timestamp"
```

### Alerting:
- **Critical:** >5% null values in key features → Page on-call
- **Warning:** Data delayed >6 hours → Slack alert
- **Info:** Schema change detected → Email data team

---

##  Success Metrics for Data Collection

**Week 4 Checkpoint:**
-  100% of customers have complete profiles
-  Engagement scores available for 95%+ customers
-  Historical data loaded (12 months)
-  Data quality dashboard live

**Week 8 Checkpoint:**
-  Feature store operational
-  Models trained and validated
-  NBA recommendations generated
-  Ready for A/B test

**Week 20 Checkpoint:**
-  A/B test data collected
-  Statistical significance achieved
-  Uplift models trained
-  Production deployment ready

---

##  FAANG Companies' Actual Practices

### Meta (Facebook):
- **Feature Store:** Custom-built (Hive/Presto)
- **A/B Platform:** In-house (Gatekeeper)
- **Data Volume:** Billions of events/day
- **Update Frequency:** Real-time streaming
- **Experimentation:** 10,000+ experiments running concurrently

### Amazon:
- **Feature Store:** SageMaker Feature Store
- **A/B Platform:** Weblab
- **Data Volume:** Petabytes
- **Update Frequency:** Batch (hourly for most ML features)
- **Experimentation:** Thousands of experiments, very rigorous

### Uber:
- **Feature Store:** Michelangelo (in-house)
- **A/B Platform:** ExpL
- **Data Volume:** Terabytes/day
- **Update Frequency:** Mixed (streaming for real-time, batch for heavy compute)
- **Experimentation:** Focus on causal inference

### Netflix:
- **Feature Store:** Custom
- **A/B Platform:** ABTest
- **Data Volume:** Terabytes/day
- **Update Frequency:** Batch (daily for most)
- **Experimentation:** Extremely sophisticated, causal ML

**Common Pattern:**
All use → Historical data for initial models → Large-scale A/B tests → Causal/uplift models → Continuous learning

---

##  Final Checklist: Data Readiness

Before launching your NBA system, ensure:

**Infrastructure:**
- [ ] Data warehouse accessible
- [ ] Feature store configured
- [ ] Streaming pipeline (if needed) operational
- [ ] ML serving platform ready
- [ ] A/B testing framework integrated

**Data Availability:**
- [ ] Customer profiles (daily batch)
- [ ] Engagement signals (real-time or hourly)
- [ ] Churn scores (weekly)
- [ ] Historical actions/outcomes (if any)
- [ ] Revenue/contract data

**Data Quality:**
- [ ] <2% missing values in critical fields
- [ ] Validation rules enforced
- [ ] Monitoring dashboards live
- [ ] Alert system configured

**Compliance:**
- [ ] PII encryption
- [ ] Access controls (RBAC)
- [ ] Audit logging
- [ ] GDPR/UAE compliance check
- [ ] Data retention policies

**Team Alignment:**
- [ ] Data sources documented
- [ ] Collection scripts in version control
- [ ] On-call rotation defined
- [ ] Runbooks for common issues

---

**When all checks pass → You're ready to build!** 
