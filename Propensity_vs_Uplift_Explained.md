# Propensity vs Uplift: The Critical Difference

##  The Core Question

### Propensity Model Asks:
**"Who's most likely to respond?"**

### Uplift Model Asks:
**"Who benefits MOST from our intervention?"**

**Why this matters:** The answer to these questions can be COMPLETELY DIFFERENT!

---

## Concrete Example: Four Customer Types

Let's say we're deciding whether to offer a retention discount to save customers from churning.

### The Four Types of Customers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WILL THEY STAY?                              │
│                                                                 │
│              WITHOUT Discount    WITH Discount    Uplift        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sure Things       95%               98%        +3%             │
│    (Already happy)                                              │
│    → Stay anyway, waste discount on them                        │
│                                                                 │
│  Persuadables      40%               85%        +45%            │
│    (On the fence)                                               │
│    → THESE are who we want to target!                           │
│                                                                 │
│  Lost Causes       5%                10%        +5%             │
│    (Deeply unhappy)                                             │
│    → Will churn anyway, waste resources                         │
│                                                                 │
│  Sleeping Dogs     80%               50%        -30%            │
│    (Happy but...) Offering discount reminds them to shop around!│
│    → ACTIVELY HARMFUL to contact!                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

##  What Each Model Predicts

### Customer Profile Example:
```python
customer_alice = {
    'tenure': 48,
    'revenue': 100000,
    'engagement': 95,
    'support_tickets': 0,
    'nps_score': 9
}
```

### Propensity Model Says:
```python
P(Retain) = 0.95  # 95% likely to stay

# Propensity ranks Alice as HIGH priority
# Reasoning: "She's very likely to respond positively!"
# Model sees: High engagement, no complaints → Will retain
```

### Uplift Model Says:
```python
P(Retain | Discount given) = 0.98   # 98% if we offer discount
P(Retain | No discount)    = 0.95   # 95% if we do nothing
Uplift = 0.98 - 0.95 = +0.03       # Only 3% incremental benefit!

# Uplift ranks Alice as LOW priority
# Reasoning: "She'll stay anyway, save the discount!"
# Model sees: She's a 'Sure Thing' - wasted investment
```

**THE PARADOX:** 
-  Propensity says: TARGET Alice (high response)
-  Uplift says: SKIP Alice (low incremental value)

**Who's right?** UPLIFT! Because Alice will stay without the discount.

---

##  Business Impact: Real Numbers

### Scenario: 1,000 customers in renewal window
- Budget: $50,000 (can offer 1,000 × $50 discount OR 100 × $500 calls)
- Revenue per customer: $100,000/year

### Strategy 1: Target by PROPENSITY (V1)

| Customer Type | Count | Propensity Score | Targeted? | Outcome Without | Outcome With | Incremental Wins | Cost | ROI |
|---------------|-------|------------------|-----------|-----------------|--------------|------------------|------|-----|
| Sure Things | 300 | 0.95 (HIGH) |  Yes | 285 stay | 294 stay | **+9** | $15,000 | **-$85K**  |
| Persuadables | 200 | 0.50 (MEDIUM) |  No | 80 stay | 170 stay | **0** (missed!) | $0 | $0 |
| Lost Causes | 200 | 0.10 (LOW) |  No | 10 stay | 20 stay | **0** (missed) | $0 | $0 |
| Sleeping Dogs | 300 | 0.85 (HIGH) |  Yes | 240 stay | 150 stay | **-90** | $15,000 | **-$915K**  |

**TOTAL PROPENSITY STRATEGY:**
- Spend: $30,000
- Incremental customers saved: 9 - 90 = **-81** (LOST customers!)
- Revenue impact: **-$8.1M** (disaster!)
- Why? Wasted money on Sure Things, annoyed Sleeping Dogs

---

### Strategy 2: Target by UPLIFT (V2)

| Customer Type | Count | Uplift Score | Targeted? | Outcome Without | Outcome With | Incremental Wins | Cost | ROI |
|---------------|-------|--------------|-----------|-----------------|--------------|------------------|------|-----|
| Sure Things | 300 | +0.03 (LOW) |  No | 285 stay | 294 stay | **0** (saved $15K) | $0 | $0 |
| Persuadables | 200 | +0.45 (HIGH) |  Yes | 80 stay | 170 stay | **+90** | $10,000 | **+$890K**  |
| Lost Causes | 200 | +0.05 (LOW) |  No | 10 stay | 20 stay | **0** (saved $10K) | $0 | $0 |
| Sleeping Dogs | 300 | -0.30 (NEGATIVE) |  No | 240 stay | 150 stay | **0** (avoided harm!) | $0 | $0 |

**TOTAL UPLIFT STRATEGY:**
- Spend: $10,000
- Incremental customers saved: **+90**
- Revenue impact: **+$9M** (massive win!)
- Efficiency: 3x less spend, 10x better outcome!

---

##  The Math Behind It

### Propensity Model (Single Prediction):
```python
# Propensity predicts ABSOLUTE probability
propensity_model.predict(customer_features)
→ Output: P(Positive Outcome) = 0.85

# Question it answers: "How likely is a good outcome?"
# Missing: "Would it happen WITHOUT our action?"
```

**Formula:**
$$P(Y=1|X) = \text{Propensity Score}$$

**Training Data:** All past customers with outcomes
**Problem:** Includes people who would have succeeded anyway!

---

### Uplift Model (Differential Prediction):
```python
# Uplift predicts DIFFERENCE between scenarios
treatment_model.predict(customer_features) → 0.85
control_model.predict(customer_features)   → 0.80
uplift = 0.85 - 0.80 = 0.05

# Question it answers: "How much EXTRA benefit from action?"
# Captures: TRUE incremental impact
```

**Formula:**
$$\tau(X) = E[Y|X, T=1] - E[Y|X, T=0] = \text{Uplift}$$

**Training Data:** Randomized A/B test with treatment AND control groups
**Advantage:** Separates natural outcomes from treatment effects!

---

##  Real-World Analogy

### Imagine you're a doctor treating headaches:

**Propensity Approach:**
- Track: "Who gets better?"
- Observe: Athletes and young people recover at 95% rate
- Conclude: "Prescribe medicine to athletes!" (high propensity to recover)
- Problem: **They'd recover anyway!** You're wasting medicine.

**Uplift Approach:**
- Track: "Who gets better BECAUSE of medicine?"
- Randomized trial: 
  - Athletes: 95% with medicine, 93% without → **+2% uplift**
  - Elderly: 60% with medicine, 30% without → **+30% uplift**
- Conclude: "Prescribe medicine to elderly!" (high incremental benefit)
- Success: **Maximum lives saved per medicine unit**

---

##  Visual Representation

### Propensity Model Rankings:
```
High Propensity → Target First
│
├─  Sure Things (0.95)      ← WASTES MONEY
├─  Sleeping Dogs (0.85)    ← ACTIVELY HARMFUL!
├─  Persuadables (0.50)     ← Missed opportunity
└─  Lost Causes (0.10)      ← Correctly skipped

Result: Target wrong customers!
```

### Uplift Model Rankings:
```
High Uplift → Target First
│
├─  Persuadables (+0.45)    ← CORRECT TARGET!
├─  Lost Causes (+0.05)     ← Correctly skipped
├─  Sure Things (+0.03)     ← Correctly skipped
└─  Sleeping Dogs (-0.30)   ← Correctly avoided!

Result: Maximum ROI!
```

---

##  Side-by-Side Comparison

### Example Scenario: Retention Campaign

| Aspect | Propensity Model | Uplift Model |
|--------|------------------|--------------|
| **Question** | "Who will stay if contacted?" | "Who will stay ONLY IF contacted?" |
| **Model Output** | 0.85 (85% retention) | +0.15 (15% incremental lift) |
| **Interpretation** | "High chance of success!" | "15% more likely vs doing nothing" |
| **Business Decision** | Contact them | Calculate ROI: 0.15 × $100K = $15K vs $500 cost |
| **Can identify "Sure Things"?** |  No - looks like everyone else with high score |  Yes - low uplift score |
| **Can identify "Sleeping Dogs"?** |  No - high propensity misleads |  Yes - NEGATIVE uplift score |
| **Can identify "Persuadables"?** |  Maybe, if low propensity |  Yes - high uplift score |
| **Avoids wasted spend?** |  No |  Yes |

---

##  Real Customer Examples

### Customer 1: Sarah (Sure Thing)
```python
Profile:
- Tenure: 60 months (5 years)
- NPS: 9/10
- Engagement: 95/100
- Recent activity: High

Propensity Model:
- P(Retain) = 0.98
- Ranking: TOP 1% (must contact!)
- Reasoning: "Extremely loyal, will definitely stay!"

Uplift Model:
- P(Retain | Call) = 0.99
- P(Retain | No call) = 0.98
- Uplift = +0.01
- Ranking: BOTTOM 20% (skip!)
- Reasoning: "Already staying, save the $500 call cost"

Business Decision:
-  SKIP (following uplift)
- Outcome: Sarah renews anyway, save $500 ✓
- Propensity would have wasted $500!
```

---

### Customer 2: Mike (Persuadable)
```python
Profile:
- Tenure: 18 months
- NPS: 6/10 (neutral)
- Engagement: 55/100
- Recent activity: Declining
- Support tickets: 3 (moderate issues)

Propensity Model:
- P(Retain) = 0.45
- Ranking: MIDDLE 50% (low priority)
- Reasoning: "Only 50-50 chance, might skip"

Uplift Model:
- P(Retain | Call) = 0.80
- P(Retain | No call) = 0.35
- Uplift = +0.45
- Ranking: TOP 5% (definitely contact!)
- Reasoning: "45% boost from intervention - huge ROI!"

Business Decision:
-  CONTACT (following uplift)
- Outcome: Mike agrees to stay after call
- ROI: 0.45 × $100K = $45K value from $500 call = 90x ROI!
- Propensity would have MISSED this opportunity!
```

---

### Customer 3: Lisa (Sleeping Dog)
```python
Profile:
- Tenure: 36 months
- NPS: 8/10
- Engagement: 70/100
- Recent activity: Steady
- Price-sensitive industry

Propensity Model:
- P(Retain) = 0.80
- Ranking: TOP 20% (should contact)
- Reasoning: "Good retention likelihood"

Uplift Model:
- P(Retain | Discount offer) = 0.50
- P(Retain | No offer) = 0.80
- Uplift = -0.30 (NEGATIVE!)
- Ranking: BOTTOM 1% (DO NOT CONTACT!)
- Reasoning: "Offering discount signals desperation, triggers shopping around"

Business Decision:
-  DO NOT CONTACT (following uplift)
- Outcome: Lisa renews at full price 
- Propensity would have:
  - Sent discount offer
  - Lisa thinks: "If they're offering discount, maybe I'm overpaying?"
  - Lisa shops around 
  - Lisa finds competitor 
  - Lost $100K customer! 
```

---

### Customer 4: David (Lost Cause)
```python
Profile:
- Tenure: 6 months
- NPS: 2/10 (detractor)
- Engagement: 10/100
- Support tickets: 15 (many complaints)
- Competitor evaluation: Yes

Propensity Model:
- P(Retain) = 0.05
- Ranking: BOTTOM 5% (skip)
- Reasoning: "Very unlikely to stay"

Uplift Model:
- P(Retain | Call + Offer) = 0.10
- P(Retain | No action) = 0.05
- Uplift = +0.05
- Ranking: LOW (but positive - weak target)
- Reasoning: "5% boost, but low absolute value"

Business Decision:
-  BOTH AGREE: Skip (unless desperate)
- ROI calculation: 0.05 × $100K = $5K vs $500 cost = 10x ROI
- Edge case: Technically worth it, but low confidence
- Better to focus on higher uplift customers
```

---

##  Key Insights Summary

### 1. **Propensity ≠ Value**
High propensity doesn't mean high value. Sure Things have high propensity but LOW incremental value.

### 2. **The Four Quadrants**

```
                  High Uplift  │  Low Uplift
                               │
    High          PERSUADABLES │   SURE THINGS
  Propensity     (TARGET!)     │  (Skip - waste)
                               │
    ─────────────┼──────────────┼─────────────
                               │
    Low           LOST CAUSES  │   SLEEPING DOGS
  Propensity     (Weak target) │  (AVOID - harm!)
                               │
```

**Propensity focuses on the LEFT column only**
**Uplift optimizes across ALL quadrants**

### 3. **Why Propensity Fails**

Propensity model cannot distinguish:
-  High score because action works? OR
-  High score because they'd succeed anyway?

**Without a control group, you can't tell!**

### 4. **Why Uplift Succeeds**

Uplift model learns from A/B test:
-  Sees outcomes WITH action (treatment group)
-  Sees outcomes WITHOUT action (control group)
-  Calculates TRUE incremental effect

**With both groups, you isolate the treatment effect!**

---

##  Mathematical Proof

### Setup:
- Action cost: $500
- Customer value: $100,000
- Break-even uplift: $500 / $100,000 = 0.5%

### Customer A (Sure Thing):
```
Propensity: 0.95 → TOP PRIORITY
Uplift: 0.03 → SKIP

Expected Value (Propensity):
  EV = P(success) × Value - Cost
     = 0.95 × $100K - $500 = $94,500  ✓ (looks good!)

Expected Value (Uplift):
  EV = Uplift × Value - Cost
     = 0.03 × $100K - $500 = $2,500  ✓
  
  BUT! Opportunity cost:
  Without action: 0.92 × $100K = $92,000
  With action:    0.95 × $100K - $500 = $94,500
  TRUE incremental: $94,500 - $92,000 = $2,500 ✓

Verdict: Worthwhile but LOW priority (better opportunities exist)
```

### Customer B (Persuadable):
```
Propensity: 0.50 → LOW PRIORITY
Uplift: 0.45 → TOP PRIORITY

Expected Value (Propensity):
  EV = 0.50 × $100K - $500 = $49,500  ✓ (seems marginal)

Expected Value (Uplift):
  EV = 0.45 × $100K - $500 = $44,500  ✓
  
  Opportunity cost:
  Without action: 0.05 × $100K = $5,000
  With action:    0.50 × $100K - $500 = $49,500
  TRUE incremental: $49,500 - $5,000 = $44,500 ✓

Verdict: HIGHEST incremental value - definitely target!
```

**Conclusion:** Customer B has 18x higher incremental value than Customer A, but propensity ranked A higher!

---

##  Production Implications

### With Propensity Models (V1):
```python
# Simple scoring
for customer in customers:
    score = propensity_model.predict(customer)
    if score > 0.7:
        take_action(customer)

# Result: Waste budget on Sure Things, miss Persuadables
```

### With Uplift Models (V2):
```python
# Uplift-based prioritization
for customer in customers:
    uplift = uplift_model.predict(customer)
    incremental_value = uplift * customer.ltv
    roi = incremental_value / action_cost
    
    if roi > 10:  # 10x ROI threshold
        take_action(customer)

# Result: Maximum incremental value per dollar spent
```

### Budget Allocation:
```python
# V1: Propensity (naive)
budget = $100,000
customers_sorted = sort_by_propensity(all_customers)
target_top_n = budget / cost_per_action
# → Targets many Sure Things (wasted)

# V2: Uplift (optimal)
budget = $100,000
customers_sorted = sort_by_uplift(all_customers)
target_while(incremental_value > cost):
    if uplift[i] * ltv[i] > cost:
        target(customer[i])
# → Targets only high-ROI customers (efficient)
```

---

##  When to Use What

### Use Propensity Models When:
-  You don't have A/B test data yet
-  You want to understand baseline likelihood
-  You're building V1.0 (initial system)
-  Budget constraints aren't tight (can afford waste)

### Use Uplift Models When:
-  You have A/B test data (randomized outcomes)
-  Budget is constrained (must optimize spend)
-  You want TRUE ROI measurement
-  You need to avoid negative effects (Sleeping Dogs)
-  You're building V2.0 (production-grade)

---

##  The Fundamental Difference (Summary)

### Propensity:
**"Who's most likely to say YES?"**
- Answers: Will they respond?
- Problem: Can't tell if they'd respond anyway
- Use case: Initial screening, broad targeting

### Uplift:
**"Who says YES because of our action?"**
- Answers: Will they respond ONLY if we act?
- Problem: Requires experimental data
- Use case: Optimized targeting, budget efficiency

---

##  The Bottom Line

**Propensity tells you WHO to target if you had unlimited budget.**

**Uplift tells you WHO to target to maximize ROI with limited budget.**

In business, budget is ALWAYS limited → **Uplift wins.**

---

##  Further Reading

- **"Sleeping Dogs" phenomenon**: Research by Ascarza (2018) - Columbia Business School
- **Uplift modeling**: "Uplift Modeling and Its Implications for B2B Customer Churn" (Devriendt et al.)
- **FAANG applications**: Uber Engineering blog on causalml
- **Economic theory**: Treatment heterogeneity in applied econometrics

**Key Paper:** "Do Not Disturb: Customer-Level Evidence on the Effect of Retention Calls" - Eva Ascarza (Harvard Business Review)
- Shows that retention calls can INCREASE churn for satisfied customers
- Proves Sleeping Dogs exist in real data
- Validates need for uplift modeling
