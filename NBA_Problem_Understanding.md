# Next Best Action - Problem Understanding

## THE CORE BUSINESS PROBLEM

**Current State:** can predict churn, but doesn't know what action to take

**Key Question:** "What should we DO about at-risk customers?"

---

## THREE FUNDAMENTAL QUESTIONS

### Q1: How risky is this customer?
- Will they churn? (Already have this from existing model)

### Q2: How valuable is this customer?
- Current + future revenue (CLV)
- Cost to serve

### Q3: Will our action actually work? ← **CRITICAL**
- Not "will they respond?"
- But "will they respond BECAUSE of our action?"

---

## FOUR CUSTOMER TYPES (Uplift Framework)

1. **Sure Things** - Stay regardless of action → Don't waste money
2. **Persuadables** - Stay ONLY if we act → **TARGET THESE**
3. **Lost Causes** - Leave regardless → Don't waste effort
4. **Sleeping Dogs** - Action backfires (rare)

**Goal:** Identify Persuadables to maximize ROI

---

## THREE SOLUTION LEVELS

### Level 1: Rule-Based
```
IF churn>70% AND value>10K → Call them
IF churn>70% AND value<10K → Email discount
```
**Pro:** Simple, transparent  
**Con:** Rigid, doesn't learn

### Level 2: Propensity Scoring
- Predict P(success | action)
- Calculate: Expected Value = P(success) × Revenue - Cost
- Choose highest EV action

**Pro:** Data-driven, considers economics  
**Con:** Ignores "what if we did nothing?"

### Level 3: Uplift/Causal Modeling ← **ADVANCED**
- Predict: Effect of action vs doing nothing
- Uplift = P(success | action) - P(success | no action)
- Focus only on incremental impact

**Pro:** True causality, maximizes ROI  
**Con:** Requires experimental data, complex

---

## KEY BUSINESS TRADE-OFFS

**Retention vs Upsell Conflict:**
- Upselling can create churn risk
- Need to balance opportunity vs risk

**Resource Constraints:**
- Can't intervene on everyone
- Budget for offers, sales team time

**Measurement:**
- A/B test to prove incremental value
- Business metrics, not model accuracy

---

## EXAMPLE: COMPLETE FLOW

**Customer G:**
- Churn risk: 75%
- Value: $15K/year
- Engagement: Low
- Past: Responds to calls

**Analysis:**
- Uplift if we call: +30% retention
- Expected value: 0.3 × $15K - $200 = $4.3K
- **Action:** Account manager call

**Customer H:**
- Same risk (75%), same value ($15K)
- But: Complained 5x, rejected offers
- Uplift if we call: +5% retention
- Expected value: 0.05 × $15K - $200 = $550
- **Action:** Do nothing (cut losses)

---

## A/B TEST REQUIREMENT

**Setup:**
- Control: Business as usual
- Treatment: NBA recommendations

**Measure (3 months):**
- Retention rate difference
- Revenue per customer difference
- Cost efficiency

**Purpose:** Prove NBA causes value, not just correlates

---

