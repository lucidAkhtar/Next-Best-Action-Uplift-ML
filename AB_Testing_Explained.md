# A/B Testing in NBA: The Complete Picture

**Question:** "If we have data to build models, why A/B test? If no data, how to build models?"

**Answer:** You have **two different types** of data needs:

---

##  Type 1: Historical Data (Available NOW)

### What It Contains:
```
Customer Features         Past Actions              Outcomes
─────────────────────   ─────────────────────   ──────────────────
• Tenure: 24 months     • We called high-value   • Churned: Yes/No
• Revenue: $75K         • We didn't call low-    • Revenue: $X
• Engagement: 60        value                    • Upsell: Yes/No
• Churn Prob: 0.7       • Biased selection!      
```

### What You CAN Do:
 Build segmentation (Risk × Value × Engagement)  
 Train propensity models (who's likely to upsell)  
 Create recommendation logic  
 Calculate expected value estimates

### What You CANNOT Do:
 **Know counterfactuals** - "What WOULD have happened if we took a different action?"  
 **Trust observed outcomes** - Selection bias contaminates everything  
 **Measure true incremental lift** - Can only see correlation, not causation

---

##  Type 2: Experimental Data (Collect via A/B Test)

### What A/B Test Provides:
```
Randomization → Causal Inference
─────────────────────────────────────────────────────────────
CONTROL Group (30%):     Business as usual
TREATMENT Group (70%):   Follow NBA recommendations

Compare outcomes → TRUE incremental impact
```

### Dual Purpose of A/B Test:

#### **Purpose 1: VALIDATION** 
*"Do our initial models actually work better than status quo?"*

**From Our Simulation:**
- Control Retention: 57.1%
- Treatment Retention: 61.7%
- **Lift: +8.1%**  Statistically significant (p=0.03)
- **Business Impact:** $1.07M annually

** VERDICT:** Initial models work! Safe to deploy.

#### **Purpose 2: DATA COLLECTION**
*"Collect gold-standard data to build even better models"*

**What We Collect:**
```python
For each customer:
├── Recommended Action: "retention_call"
├── Action Taken: "retention_call" (or override)
├── Customer Features: tenure=24, revenue=$75K...
├── Outcome (Treatment): Retained? Revenue?
└── Counterfactual (Control): What happened without action?
    └── This is IMPOSSIBLE to get from historical data!
```

**New Capability Unlocked:** **Uplift Modeling**

---

##  The Selection Bias Problem (Why You Need A/B)

### Historical Data Shows Paradox:
```
Our simulation results:
├─ Customers with NO action:     36.6% churn rate
└─ Customers with CALLS:          43.6% churn rate

 Does this mean calls INCREASE churn?! 
```

**NO!** We called the RISKIEST customers (selection bias).

### A/B Test Reveals Truth:
```
Randomized comparison:
├─ Control (no systematic intervention):  42.9% churn
└─ Treatment (NBA recommendations):       38.3% churn

 Actions REDUCE churn by 10.7%!
```

**This is why FAANG companies ALWAYS A/B test before scaling.**

---

##  The Complete Pipeline (FAANG Standard)

### PHASE 1: Historical Analysis (Week 1-8)
```python
# INPUT: Historical observational data
historical_data = {
    'customer_id': [1, 2, 3, ...],
    'tenure_months': [24, 36, 12, ...],
    'churn_probability': [0.7, 0.3, 0.5, ...],  # From existing model
    'engagement_score': [60, 85, 45, ...],
    'annual_revenue': [75000, 15000, 50000, ...]
}

# ANALYSIS
segments = create_3d_segmentation(historical_data)
upsell_model = train_propensity_model(historical_data)
nba_recommendations = assign_actions(segments, upsell_model)

# OUTPUT: Initial NBA system (NOT validated yet)
```

**Assumption at this stage:** Our logic is sound, but UNPROVEN.

---

### PHASE 2: A/B Test (Week 9-20)
```python
# SETUP
renewal_customers = get_customers_in_renewal_window()
control, treatment = random_split(renewal_customers, ratio=0.3/0.7)

# EXECUTION
control.apply_action('business_as_usual')      # Current approach
treatment.apply_action(nba_recommendations)    # Our new system

# COLLECT (This is the KEY data!)
ab_results = pd.DataFrame({
    'customer_id': [...],
    'group': ['control', 'treatment', ...],
    'features': [...],
    'action_taken': ['none', 'retention_call', ...],
    'outcome_retained': [1, 0, 1, ...],
    'revenue': [75000, 0, 90000, ...],
    'cost': [0, 500, 0, ...]
})

# MEASURE
retention_lift = (treatment.retention - control.retention) / control.retention
revenue_lift = (treatment.revenue - control.revenue) / control.revenue
statistical_test(retention_lift)  # p < 0.05? Significant!
```

**Output:**
1.  Proof our system works (validation)
2.  Rich experimental dataset (for uplift models)

---

### PHASE 3: Uplift Modeling (Week 21-26)
```python
# NOW we can train uplift models!
# This was IMPOSSIBLE before A/B test

# Two-Model Approach
treatment_data = ab_results[ab_results['group'] == 'treatment']
control_data = ab_results[ab_results['group'] == 'control']

# Model 1: P(Retain | Treated)
model_treat = train(treatment_data.features, treatment_data.retained)

# Model 2: P(Retain | Not Treated)  
model_control = train(control_data.features, control_data.retained)

# Uplift Score
def predict_uplift(customer):
    p_treat = model_treat.predict(customer)      # 62% retention if we act
    p_control = model_control.predict(customer)  # 59% retention if we don't
    uplift = p_treat - p_control                 # 3% incremental benefit
    return uplift

# Decision Rule
if predict_uplift(customer) > threshold:
    recommend_action()  # Worth it!
else:
    save_resources()     # Low incremental benefit
```

**Why This Is Better:**
- Old propensity model: "Who's likely to respond?"  (might respond anyway)
- New uplift model: "Who ONLY responds because of our action?"  (true lift)

**FAANG Example:**
- Netflix: "Which customers need a discount to stay?" (not who's price-sensitive)
- Uber: "Which drivers need bonus to stay online?" (not who works a lot anyway)
- Amazon: "Which customers will buy MORE with Prime?" (not who shops a lot already)

---

##  Real Numbers from Our Simulation

### What We Started With (Historical):
| Metric | Value | Issue |
|--------|-------|-------|
| Customers | 10,000 |  Plenty of data |
| Actions taken | 3,523 |  Biased (only high-risk) |
| Paradox | Calls → Higher churn! |  Selection bias |

### What We Learned (A/B Test):
| Metric | Control | Treatment | Lift |
|--------|---------|-----------|------|
| Retention | 57.1% | 61.7% | **+8.1%**  |
| Avg Revenue | $16,564 | $16,666 | +0.6% |
| Annual Impact | - | - | **+$1.07M** |
| Significance | - | - | p=0.03  |

### What We Built (Uplift Models):
```
Example Customer:
├─ Churn Probability: 70%
├─ Revenue: $75,000
├─ Engagement: 60
│
├─ Model Prediction:
│   ├─ WITH action:     62% retention
│   └─ WITHOUT action:  59% retention
│   └─ UPLIFT:          +3% (incremental benefit)
│
└─ Decision: Uplift < 5% threshold → DON'T intervene (save resources)
```

---

##  Key Takeaways (FAANG Perspective)

### 1. **Historical Data ≠ Experimental Data**
- Historical: Observational, biased, correlation
- Experimental: Randomized, unbiased, causation
- **Need BOTH** for robust system

### 2. **A/B Testing Has Dual Purpose**
- Validation: "Does it work?"
- Data Collection: "Build better models"
- **Not optional** for production ML

### 3. **The Timeline**
```
Week 1-8:    Build Version 1.0 (propensity models)
Week 9-20:   A/B test (validate + collect)
Week 21-26:  Build Version 2.0 (uplift models)
Quarter 2+:  Continuous learning loop
```

### 4. **Why This Is Standard at FAANG**
Every major tech company follows this:
- **Meta:** News feed ranking → A/B → Uplift modeling
- **Google:** Ad recommendations → A/B → Causal models  
- **Amazon:** Product recommendations → A/B → Personalization
- **Netflix:** Content recommendations → A/B → Treatment effects

**Core Principle:** "In God we trust, all others bring data." - W. Edwards Deming

### 5. **What Makes This Senior-Level Thinking**
Junior: "Build a model with available data"  
Senior: "Use available data for V1, design experiments to collect the RIGHT data for V2"

---

##  Sample Files Generated

Check these files to see the actual data:

1. **[historical_data_with_nba.csv](historical_data_with_nba.csv)**
   - 10,000 customers with features, segments, NBA recommendations
   - Use this to understand "before A/B" state

2. **[ab_test_results.csv](ab_test_results.csv)**
   - 2,621 customers in A/B test with outcomes
   - Use this to see experimental data structure

3. **[nba_data_pipeline_example.py](nba_data_pipeline_example.py)**
   - Complete pipeline code
   - Run to simulate full flow

---

##  Your Original Question Answered

**"How does data collection happen for successful delivery?"**

### Data Collection Timeline:

**Day 1 (Historical Data - Already Have):**
```python
✓ Customer attributes (CRM)
✓ Engagement signals (product analytics)
✓ Churn scores (existing model)
✓ Past actions/outcomes (biased!)
```
**→ Build initial NBA system**

**Week 9-20 (A/B Test - Collect NOW):**
```python
✓ Randomized treatment assignment
✓ Action taken + customer features
✓ Outcomes for treatment group
✓ Outcomes for control group (counterfactuals!)
```
**→ Validate system + collect gold-standard data**

**Week 21+ (Continuous - Ongoing):**
```python
✓ All production recommendations logged
✓ Outcomes tracked (30/60/90 days)
✓ Monthly model retraining
✓ Quarterly segment reviews
```
**→ Continuous improvement**

---

**Bottom Line:** You DON'T need A/B test data to START. You DO need it to PROVE and IMPROVE. That's why the timeline is phased: Build → Test → Learn → Rebuild.
