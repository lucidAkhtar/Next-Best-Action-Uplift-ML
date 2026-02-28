# Quick Reference: Your A/B Test Questions Answered

## Your Original Questions

### Q1: "What is upsell propensity? What is treatment response?"

**Upsell Propensity Model:**
- **Predicts:** P(Customer accepts upsell offer)
- **Input:** Engagement, feature adoption, tenure, revenue
- **Output:** Score 0-1 (e.g., 0.6 = 60% likely to accept)
- **Use:** Identify customers receptive to growth opportunities
- **Example:** Customer with 95% engagement + 36 months tenure = 0.8 propensity

**Treatment Response Model:**
- **Predicts:** Which channel works best (call vs email vs offer)
- **Input:** Customer characteristics + historical response patterns
- **Output:** Optimal action channel
- **Use:** Resource optimization (don't waste expensive calls on email-responsive customers)
- **Example:** High-value + senior decision-maker = "retention_call" more effective

---

### Q2: "How does data collection happen and what data to collect?"

**Data You HAVE (Start with this):**
```
Historical Observational Data:
├─ Customer profiles (CRM)
├─ Engagement metrics (Product analytics)
├─ Churn scores (Existing model)
├─ Past actions/outcomes (With selection bias!)
└─ USE: Build initial NBA system (V1.0)
```

**Data You COLLECT (via A/B test):**
```
Experimental Causal Data:
├─ Randomized assignments (control vs treatment)
├─ Actions taken for BOTH groups
├─ Outcomes for BOTH groups
├─ Counterfactuals (what would have happened)
└─ USE: Validate V1.0 + Build V2.0 (uplift models)
```

**See files for detailed specs:**
- [Data_Collection_FAANG_Standards.md](Data_Collection_FAANG_Standards.md) - Complete data requirements
- [ab_test_results.csv](ab_test_results.csv) - Example experimental data
- [historical_data_with_nba.csv](historical_data_with_nba.csv) - Example historical data

---

### Q3: "A/B test is for collecting data but its usage is to understand performance, right?"

**BOTH! A/B test serves TWO purposes:**

#### Purpose 1: VALIDATION (Understand Performance)
```
Question: "Does our NBA system work better than current approach?"

Method:
├─ Control: Business as usual (57% retention)
├─ Treatment: NBA recommendations (62% retention)
└─ Compare: +8% lift, p=0.03 (significant!)

Answer: YES! Safe to deploy.
```

#### Purpose 2: DATA COLLECTION (Enable Better Models)
```
Question: "Collect gold-standard data for causal modeling"

What We Get:
├─ Randomized outcomes (removes bias)
├─ Counterfactuals (what WOULD have happened)
├─ Heterogeneous effects (who benefits most)
└─ Use: Build uplift models predicting INCREMENTAL impact

New Capability: Uplift modeling (impossible with historical data alone)
```

**Timeline:**
- Week 9-12: See if it works (validation)
- Week 13-20: Continue collecting data
- Week 21+: Use collected data to build uplift models

---

##  The Complete Picture (One-Page Visual)

```
STARTING POINT: Historical Data
─────────────────────────────────────────────────────
CRM Data: 10,000 customers, 12 months history
Engagement: Login frequency, feature usage
Churn Model: Weekly churn probability scores
Past Actions: Previous interventions (biased!)

PROBLEM: Selection bias, no counterfactuals
CAN DO: Build initial models, segmentation


PHASE 1 (Week 1-8): Build Initial NBA System
─────────────────────────────────────────────────────
1. Do EDA:
   ├─ Understand churn patterns by segment
   ├─ Analyze past intervention effectiveness
   └─ Identify high-value, at-risk customers

2. Create 3D Segmentation:
   ├─ Risk: High/Medium/Low (from churn model)
   ├─ CLV: High/Medium/Low (revenue × retention × time)
   └─ Engagement: High/Low (product usage score)

3. Build Propensity Models:
   ├─ Upsell Propensity: P(accept upsell)
   ├─ Training data: Past upsell attempts/outcomes
   └─ Model: XGBoost on engagement + tenure

4. Design NBA Logic:
   ├─ IF risk >60% AND value >50K: retention_call
   ├─ IF risk >60% AND value <50K: retention_email
   ├─ IF risk <20% AND engagement >75: upsell_offer
   └─ ELSE: none (monitor only)

5. Calculate Expected Value:
   EV = P(Success) × Incremental_Revenue - Cost

OUTPUT: NBA recommendations for all customers
GAP: Don't know if this ACTUALLY works!


PHASE 2 (Week 9-20): A/B Test → Dual Purpose
─────────────────────────────────────────────────────
 SETUP:
├─ Sample: 2,500 customers in renewal window
├─ Control (30%): Business as usual
├─ Treatment (70%): Follow NBA recommendations
└─ Stratify: By segment to ensure balance

 COLLECT (Every customer in test):
├─ Pre-test features (risk, value, engagement)
├─ Assignment (control/treatment)
├─ Action taken (what we did)
├─ Cost incurred ($500 call, $50 email)
├─ Outcome @ 30 days (immediate response)
├─ Outcome @ 90 days (churned or retained)
└─ Revenue impact (total value)

 PURPOSE 1 - VALIDATION:
   Control Retention: 57%
   Treatment Retention: 62%
   ➜ LIFT: +8% (p=0.03)  IT WORKS!
   ➜ DECISION: Safe to deploy to production

 PURPOSE 2 - DATA COLLECTION:
   ➜ Now have CAUSAL DATA with counterfactuals
   ➜ Can see "what would have happened" (control group)
   ➜ Critical for uplift modeling


PHASE 3 (Week 21-26): Uplift Models → V2.0
─────────────────────────────────────────────────────
 LEARN from A/B test data:

1. Train Two Models:
   ├─ Model A: P(Retain | Treatment)
   │   └─ Trained on treatment group data
   │
   └─ Model B: P(Retain | Control)  
       └─ Trained on control group data

2. Predict Uplift:
   For new customer:
   ├─ Model A: "62% retention if we act"
   ├─ Model B: "59% retention if we don't"
   └─ UPLIFT: 3% incremental benefit
       └─ DECISION: 3% × $75K = $2,250 benefit
           vs $500 cost = Worth it!

3. Why This Is Better:
    OLD (Propensity): "Who's likely to respond?"
      └─ Problem: They might respond anyway!
   
    NEW (Uplift): "Who ONLY responds because of our action?"
      └─ True incremental impact

OUTPUT: Much more precise targeting
GAP: None! Ready for production


PHASE 4 (Ongoing): Continuous Learning
─────────────────────────────────────────────────────
 Production Data Collection:
├─ Log every recommendation
├─ Track every outcome
├─ Monitor model performance
├─ Retrain monthly
└─ A/B test major changes

 Feedback Loop:
   New Data → Retrain Models → Better Recommendations
   → Better Outcomes → New Data → ...

 Always Be Testing:
├─ New segments to target
├─ New channels to try
├─ New offers to test
└─ Continuous optimization
```

---

##  Files Generated for You

1. **[nba_data_pipeline_example.py](nba_data_pipeline_example.py)**
   - Complete working code showing all 4 phases
   - Run it: `python nba_data_pipeline_example.py`
   - Generated synthetic data matching FAANG patterns

2. **[AB_Testing_Explained.md](AB_Testing_Explained.md)**
   - Deep dive into why A/B testing is essential
   - Selection bias problem explained
   - Historical vs experimental data

3. **[Data_Collection_FAANG_Standards.md](Data_Collection_FAANG_Standards.md)**
   - Exact data requirements (with schemas)
   - Collection infrastructure (pipelines, feature store)
   - Data volume estimates and costs
   - Quality checks and monitoring

4. **[inspect_data.py](inspect_data.py)**
   - Shows what experimental data looks like
   - Concrete examples of customers in control vs treatment
   - Run it: `python inspect_data.py`

5. **[ab_test_results.csv](ab_test_results.csv)**
   - 2,621 customers with actual A/B test outcomes
   - Use for understanding data structure

6. **[historical_data_with_nba.csv](historical_data_with_nba.csv)**
   - 10,000 customers with features + NBA recommendations
   - Use for training initial models

---

##  Your Next Steps

1. **Understand the flow:** Read [AB_Testing_Explained.md](AB_Testing_Explained.md)

2. **See it in action:** Run `python nba_data_pipeline_example.py`

3. **Study the data:** Run `python inspect_data.py` to see examples

4. **Plan your implementation:** Use [Data_Collection_FAANG_Standards.md](Data_Collection_FAANG_Standards.md)

5. **Reference your case study:** Apply to [Component_1_Compressed_2Page.md](Component_1_Compressed_2Page.md)

---

##  Key Insight (Most Important Takeaway)

```
You DON'T need perfect data to START
You DO need good data to PROVE and IMPROVE

The magic is in the SEQUENCE:

Historical Data (Available) 
    → Initial Models (V1.0)
        → A/B Test (Validate + Collect)
            → Uplift Models (V2.0)
                → Production (Continuous Learning)

This is how EVERY FAANG company does it:
- Meta's news feed ranking
- Amazon's product recommendations  
- Netflix's content recommendations
- Uber's driver incentives
- Google's ad targeting

ALWAYS: Build → Test → Learn → Rebuild
```

---

##  Additional Resources

**If you want to dive deeper:**

- **Causal Inference:** "The Book of Why" by Judea Pearl
- **Uplift Modeling:** "Causal Machine Learning" (Microsoft Research)
- **A/B Testing:** "Trustworthy Online Controlled Experiments" (Kohavi et al.)
- **FAANG Practices:**
  - Meta: [Loss Aversion paper](https://research.facebook.com/publications/loss-aversion/)
  - Uber: [Experimentation Platform blog](https://eng.uber.com/experimentation-platform/)
  - Netflix: [A/B Testing blog series](https://netflixtechblog.com/tagged/ab-testing)

---