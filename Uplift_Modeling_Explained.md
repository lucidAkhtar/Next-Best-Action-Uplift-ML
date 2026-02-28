# Uplift Modeling: From V1 (Propensity) to V2 (Causal)

##  The Journey You Described (Perfect Understanding!)

```
V1.0: Propensity Models (Week 1-8)
├─ Built with: Historical data
├─ Predicts: P(Customer responds to action)
├─ Problem: Can't distinguish "responds BECAUSE of action" vs "would respond anyway"
└─ Tools: sklearn, XGBoost (standard supervised learning)

↓ A/B Test (Week 9-20)

A/B Test: Dual Purpose
├─ Purpose 1: Validate V1.0 performance (does it beat baseline?)
├─ Purpose 2: Collect RANDOMIZED data
│   ├─ Treatment group outcomes (with action)
│   ├─ Control group outcomes (without action)
│   └─ KEY: Same customer types in both groups (randomization)
└─ Output: Causal dataset with counterfactuals

↓

V2.0: Uplift Models (Week 21+)
├─ Built with: Experimental data (A/B test results)
├─ Predicts: Uplift = Revenue(with action) - Revenue(without action)
├─ Answers: "Who benefits MOST from intervention?"
└─ Tools: causalml, scikit-uplift, OR sklearn with special methodology
```

---

##  What DEFINES an Uplift Model?

### Traditional ML (Propensity):
```python
# Predicts ABSOLUTE probability
P(Customer buys) = 0.7

# Problem: Would they buy anyway?
# Can't tell from this alone!
```

### Uplift ML (Causal):
```python
# Predicts INCREMENTAL effect
P(Buy | Discount given) = 0.7
P(Buy | No discount)    = 0.6
Uplift = 0.7 - 0.6 = 0.1

# Now we know: 10% incremental lift from discount
# Decision: Worth the discount cost? 10% × $100 revenue = $10 vs $5 discount = YES!
```

**Key Insight:** Uplift models predict the **treatment effect**, not the outcome itself.

---

##  Uplift Modeling Is NOT a Single Algorithm

**Uplift modeling is a METHODOLOGY, not a specific algorithm.**

You can use:
-  **XGBoost** (most common at FAANG)
-  **Random Forest**
-  **Neural Networks**
-  **Logistic Regression**
-  **Any sklearn estimator**

The difference is HOW you use them, not WHAT you use.

---

##  Three Main Approaches to Uplift Modeling

### **Approach 1: Two-Model (T-Learner)  Most Common**

**Concept:** Train separate models on treatment and control, subtract predictions

```python
"""
Two-Model Approach (T-Learner)
Most popular at FAANG - simple, interpretable, works well
"""

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

def train_uplift_two_model(df_treatment, df_control, features):
    """
    Train two separate models and compute uplift
    
    Used by: Amazon, Uber, Netflix
    """
    # Model 1: Learn from customers who received treatment
    model_treatment = GradientBoostingClassifier(n_estimators=100, random_state=42)
    X_treat = df_treatment[features]
    y_treat = df_treatment['outcome']  # 1 = retained, 0 = churned
    model_treatment.fit(X_treat, y_treat)
    
    # Model 2: Learn from customers who did NOT receive treatment
    model_control = GradientBoostingClassifier(n_estimators=100, random_state=42)
    X_ctrl = df_control[features]
    y_ctrl = df_control['outcome']
    model_control.fit(X_ctrl, y_ctrl)
    
    return model_treatment, model_control

def predict_uplift(model_treatment, model_control, customer_features):
    """
    Predict uplift for a new customer
    """
    # What would happen if we treat them?
    p_treat = model_treatment.predict_proba(customer_features)[0, 1]
    
    # What would happen if we DON'T treat them?
    p_control = model_control.predict_proba(customer_features)[0, 1]
    
    # Incremental effect
    uplift = p_treat - p_control
    
    return {
        'p_with_action': p_treat,
        'p_without_action': p_control,
        'uplift': uplift,
        'decision': 'Act' if uplift > 0.05 else 'Skip'  # 5% threshold
    }

# Example usage
customer = pd.DataFrame({
    'churn_probability': [0.7],
    'annual_revenue': [75000],
    'engagement_score': [60],
    'tenure_months': [24]
})

result = predict_uplift(model_treatment, model_control, customer)
print(f"Uplift: {result['uplift']:.1%}")
print(f"Decision: {result['decision']}")

# If uplift = 8%:
# → 8% × $75K = $6,000 incremental value
# → Cost of call = $500
# → ROI = $6,000 / $500 = 12x → DEFINITELY act!
```

**Pros:**
-  Simple, interpretable
-  Can use any sklearn model
-  Works well in practice
-  Used by most FAANG companies

**Cons:**
-  Two models might learn different feature representations
-  Assumes treatment/control groups are perfectly comparable

---

### **Approach 2: Single Model with Treatment Flag (S-Learner)**

**Concept:** Train one model with treatment as a feature

```python
"""
Single-Model Approach (S-Learner)
Simpler but less accurate for heterogeneous effects
"""

def train_uplift_single_model(df_all, features):
    """
    Train one model with treatment indicator as feature
    """
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Add treatment flag as feature
    X = df_all[features + ['treatment_indicator']]  # treatment_indicator = 1 or 0
    y = df_all['outcome']
    
    model.fit(X, y)
    return model

def predict_uplift_single(model, customer_features):
    """
    Predict with and without treatment
    """
    # Predict WITH treatment
    customer_treated = customer_features.copy()
    customer_treated['treatment_indicator'] = 1
    p_treat = model.predict_proba(customer_treated)[0, 1]
    
    # Predict WITHOUT treatment
    customer_untreated = customer_features.copy()
    customer_untreated['treatment_indicator'] = 0
    p_control = model.predict_proba(customer_untreated)[0, 1]
    
    uplift = p_treat - p_control
    return uplift
```

**Pros:**
-  One model to maintain
-  Enforces consistent feature representation

**Cons:**
-  Might not capture heterogeneous treatment effects well
-  Model might ignore treatment flag if other features are stronger

---

### **Approach 3: Specialized Uplift Packages  Best Practice**

**Concept:** Use packages built specifically for uplift modeling

```python
"""
Specialized Uplift Libraries
Production-grade implementations with validation
"""

# ============================================================================
# OPTION A: causalml (Uber's library)
# ============================================================================
from causalml.inference.meta import BaseXRegressor, BaseRRegressor
from xgboost import XGBClassifier

# Meta-learner with XGBoost
uplift_model = BaseXRegressor(
    learner=XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
)

# Fit on A/B test data
uplift_model.fit(
    X=features,                    # Customer features
    treatment=treatment_indicator, # 1 = treated, 0 = control
    y=outcome                      # 1 = retained, 0 = churned
)

# Predict uplift (CATE = Conditional Average Treatment Effect)
uplift_scores = uplift_model.predict(X_new)


# ============================================================================
# OPTION B: scikit-uplift (Russian team, very good)
# ============================================================================
from sklift.models import SoloModel, ClassTransformation
from catboost import CatBoostClassifier

# Solo Model approach (two models internally)
uplift_solo = SoloModel(
    estimator=CatBoostClassifier(verbose=0, random_state=42)
)

uplift_solo.fit(
    X=features,
    y=outcome,
    treatment=treatment_indicator
)

uplift_scores = uplift_solo.predict(X_new)


# ============================================================================
# OPTION C: EconML (Microsoft Research)
# ============================================================================
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestClassifier

uplift_tlearner = TLearner(
    models=RandomForestClassifier(n_estimators=100, random_state=42)
)

uplift_tlearner.fit(
    Y=outcome,
    T=treatment_indicator,
    X=features
)

# Get treatment effect
uplift_scores = uplift_tlearner.effect(X_new)
```

**Package Comparison:**

| Package | Maintainer | Pros | Use Case |
|---------|-----------|------|----------|
| **causalml** | Uber | Production-tested, many algorithms |  Best for enterprise |
| **scikit-uplift** | Community | Clean API, sklearn-compatible | Good for startups |
| **EconML** | Microsoft | Rigorous theory, research-focused | Academic/research |
| **DoWhy** | Microsoft | Causal inference framework | Causal graph modeling |

**FAANG Standard: causalml (Uber's package)**

---

##  What Makes Data "Good" for Uplift Modeling?

###  Requirements for V2.0 (Uplift):
```python
# Your A/B test data structure
ab_test_data = pd.DataFrame({
    # Features (same for both groups)
    'customer_id': [1, 2, 3, 4, ...],
    'churn_probability': [0.7, 0.3, 0.8, 0.2, ...],
    'engagement_score': [60, 85, 40, 90, ...],
    'annual_revenue': [75000, 50000, 100000, 15000, ...],
    
    # Treatment assignment (CRITICAL: must be random!)
    'treatment': [1, 0, 1, 0, ...],  # 1 = treated, 0 = control
    
    # Outcome
    'retained': [1, 1, 0, 1, ...]  # 1 = retained, 0 = churned
})
```

**Key Requirements:**
1.  **Randomization**: Treatment assigned randomly (not based on features)
2.  **Both groups**: Need outcomes for treated AND control
3.  **Sufficient size**: 1,000+ per group (2,000+ total minimum)
4.  **Same features**: Control and treatment measured identically

**Why historical data doesn't work:**
```python
# Historical data (biased)
historical_data = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'risk': [0.9, 0.2, 0.8],
    'action_taken': ['call', 'none', 'email'],  #  NOT RANDOM!
    'outcome': [1, 1, 0]
})

# We called high-risk customers (selection bias)
# Can't learn true causal effect
```

---

##  Complete Working Example with Your A/B Test Data

```python
"""
Complete Uplift Modeling Pipeline
Using data from our A/B test simulation
"""

import pandas as pd
from causalml.inference.meta import BaseTRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load A/B test results
ab_data = pd.read_csv('ab_test_results.csv')

print(f"Total customers in A/B test: {len(ab_data)}")
print(f"Treatment: {(ab_data['ab_group']=='Treatment').sum()}")
print(f"Control: {(ab_data['ab_group']=='Control').sum()}")

# Prepare data for uplift modeling
features = ['churn_probability', 'annual_revenue', 'engagement_score', 
            'tenure_months', 'support_tickets', 'feature_adoption']

X = ab_data[features].fillna(0)
treatment = (ab_data['ab_group'] == 'Treatment').astype(int)
y = (1 - ab_data['churned']).values  # 1 = retained, 0 = churned

# Split for validation
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
    X, treatment, y, test_size=0.3, random_state=42, stratify=treatment
)

print(f"\nTraining set: {len(X_train)} customers")
print(f"Test set: {len(X_test)} customers")

# ============================================================================
# METHOD 1: Two-Model Approach (Manual)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: Two-Model Approach (T-Learner)")
print("="*80)

from sklearn.ensemble import GradientBoostingClassifier

# Separate training data
train_treatment = X_train[T_train == 1]
train_control = X_train[T_train == 0]
y_treatment = y_train[T_train == 1]
y_control = y_train[T_train == 0]

# Train two models
model_treat = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_control = GradientBoostingClassifier(n_estimators=100, random_state=42)

model_treat.fit(train_treatment, y_treatment)
model_control.fit(train_control, y_control)

# Predict on test set
p_treat_test = model_treat.predict_proba(X_test)[:, 1]
p_control_test = model_control.predict_proba(X_test)[:, 1]
uplift_manual = p_treat_test - p_control_test

print(f"Average Uplift: {uplift_manual.mean():.3f}")
print(f"Std Uplift: {uplift_manual.std():.3f}")
print(f"Min Uplift: {uplift_manual.min():.3f}")
print(f"Max Uplift: {uplift_manual.max():.3f}")

# ============================================================================
# METHOD 2: causalml Package (Recommended)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: causalml T-Learner (Production Grade)")
print("="*80)

uplift_model = BaseTRegressor(
    learner=XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
)

# Fit
uplift_model.fit(X=X_train, treatment=T_train, y=y_train)

# Predict CATE (Conditional Average Treatment Effect)
uplift_causalml = uplift_model.predict(X=X_test).flatten()

print(f"Average Uplift: {uplift_causalml.mean():.3f}")
print(f"Std Uplift: {uplift_causalml.std():.3f}")
print(f"Min Uplift: {uplift_causalml.min():.3f}")
print(f"Max Uplift: {uplift_causalml.max():.3f}")

# ============================================================================
# EVALUATION: Uplift Curve (like ROC curve for uplift)
# ============================================================================
print("\n" + "="*80)
print("EVALUATION: Cumulative Uplift")
print("="*80)

def evaluate_uplift(uplift_scores, treatment, outcome):
    """
    Calculate cumulative gain from targeting by uplift score
    """
    df = pd.DataFrame({
        'uplift': uplift_scores,
        'treatment': treatment,
        'outcome': outcome
    })
    
    # Sort by uplift (highest first)
    df = df.sort_values('uplift', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative metrics
    percentiles = np.arange(0.1, 1.1, 0.1)
    results = []
    
    for pct in percentiles:
        n = int(len(df) * pct)
        subset = df.head(n)
        
        # Treatment effect in this subset
        treated = subset[subset['treatment'] == 1]
        control = subset[subset['treatment'] == 0]
        
        if len(treated) > 0 and len(control) > 0:
            effect_treated = treated['outcome'].mean()
            effect_control = control['outcome'].mean()
            uplift_actual = effect_treated - effect_control
        else:
            uplift_actual = 0
        
        results.append({
            'percentile': pct,
            'uplift': uplift_actual
        })
    
    return pd.DataFrame(results)

uplift_eval = evaluate_uplift(uplift_causalml, T_test.values, y_test)
print(uplift_eval)

print("\n Interpretation:")
print("- Top 10% customers: Highest incremental benefit from action")
print("- Top 50% customers: Still positive uplift")
print("- Bottom 50%: May have lower/negative uplift (save resources!)")

# ============================================================================
# BUSINESS APPLICATION: Who to target?
# ============================================================================
print("\n" + "="*80)
print("BUSINESS APPLICATION")
print("="*80)

# Predict uplift for all future customers
future_customers = ab_data[features].fillna(0)
future_uplift = uplift_model.predict(X=future_customers).flatten()

# Create uplift-based recommendations
ab_data['uplift_score'] = future_uplift
ab_data['uplift_revenue'] = (
    ab_data['uplift_score'] * ab_data['annual_revenue'] * 2  # 2-year value
)

# Decision rule: Act if uplift revenue > 10x cost
ab_data['action_cost'] = ab_data['nba_action'].map({
    'retention_call': 500,
    'retention_email': 50,
    'upsell_offer': 100,
    'none': 0
}).fillna(0)

ab_data['roi'] = ab_data['uplift_revenue'] / (ab_data['action_cost'] + 1)
ab_data['v2_recommendation'] = np.where(
    ab_data['roi'] > 10,  # 10x ROI threshold
    ab_data['nba_action'],
    'none'
)

# Compare V1 vs V2 recommendations
print("\nV1.0 (Propensity-based) Recommendations:")
print(ab_data['nba_action'].value_counts())

print("\nV2.0 (Uplift-based) Recommendations:")
print(ab_data['v2_recommendation'].value_counts())

efficiency_gain = (
    (ab_data['v2_recommendation'] == 'none').sum() - 
    (ab_data['nba_action'] == 'none').sum()
)
print(f"\n🎯 Efficiency Gain: {efficiency_gain} fewer actions")
print(f"   Cost Savings: ${efficiency_gain * 50:,.0f} (assuming avg cost $50)")
print(f"   → Focus resources on highest-uplift customers!")

# Show examples
print("\n" + "="*80)
print("EXAMPLE CUSTOMERS")
print("="*80)

# High uplift example
high_uplift_idx = ab_data['uplift_score'].idxmax()
high_customer = ab_data.loc[high_uplift_idx]
print(f"\n HIGH UPLIFT Customer:")
print(f"   Churn Risk: {high_customer['churn_probability']:.1%}")
print(f"   Revenue: ${high_customer['annual_revenue']:,.0f}")
print(f"   Uplift Score: {high_customer['uplift_score']:.3f}")
print(f"   Uplift Revenue: ${high_customer['uplift_revenue']:,.0f}")
print(f"   V1 Recommendation: {high_customer['nba_action']}")
print(f"   V2 Recommendation: {high_customer['v2_recommendation']}")
print(f"   → DEFINITELY act on this customer!")

# Low uplift example
low_uplift_idx = ab_data['uplift_score'].idxmin()
low_customer = ab_data.loc[low_uplift_idx]
print(f"\n LOW UPLIFT Customer:")
print(f"   Churn Risk: {low_customer['churn_probability']:.1%}")
print(f"   Revenue: ${low_customer['annual_revenue']:,.0f}")
print(f"   Uplift Score: {low_customer['uplift_score']:.3f}")
print(f"   Uplift Revenue: ${low_customer['uplift_revenue']:,.0f}")
print(f"   V1 Recommendation: {low_customer['nba_action']}")
print(f"   V2 Recommendation: {low_customer['v2_recommendation']}")
print(f"   → Skip! They'll churn anyway OR retain anyway")

print("\n" + "="*80)
print(" UPLIFT MODEL READY FOR PRODUCTION")
print("="*80)
```

---

##  What Defines an Uplift Model: Summary

### **Technical Definition:**
An uplift model predicts the **Conditional Average Treatment Effect (CATE)**:

$$\tau(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]$$

Where:
- $\tau(x)$ = uplift for customer with features $x$
- $Y$ = outcome (retained/churned)
- $T$ = treatment (action taken or not)
- $X$ = customer features

### **Practical Definition:**
Uplift models answer: **"Who benefits MOST from our intervention?"**

NOT: "Who's most likely to respond?" (that's propensity)

### **What Makes It Different:**
| Aspect | Propensity Model (V1) | Uplift Model (V2) |
|--------|----------------------|-------------------|
| **Training Data** | Historical (biased) | Experimental (randomized) |
| **Predicts** | P(Outcome) | P(Outcome\|Treatment) - P(Outcome\|Control) |
| **Answers** | "Who will respond?" | "Who responds BECAUSE of action?" |
| **Algorithm** | Any sklearn model | Same models, different methodology |
| **Validation** | Standard metrics | Uplift curves, AUUC |
| **Business Value** | Moderate | High (avoids wasted spend) |

### **It's NOT:**
-  A specific algorithm (XGBoost, etc.)
-  Just scipy or sklearn
-  A statistical test

### **It IS:**
-  A **methodology** for training models
-  Using **causal inference** principles
-  Built with **standard ML libraries** + specialized packages
-  Requires **experimental data** (A/B test results)

---

##  FAANG Implementation Patterns

### **Meta (Facebook):**
- Uses custom two-model approach
- XGBoost base learners
- Deployed for ads, news feed ranking
- Billions of predictions per day

### **Uber:**
- Created **causalml** package (open source)
- Used for driver incentives, rider promotions
- T-Learner with gradient boosting
- Reduced incentive spend 30% while maintaining retention

### **Amazon:**
- Uplift for product recommendations, Prime offers
- Proprietary implementation
- Heavy use of randomized experiments
- Constant A/B testing to feed uplift models

### **Netflix:**
- Recommendation uplift (content suggestions)
- Subscription retention
- Treatment heterogeneity analysis
- Sophisticated causal ML research team

---

##  Your Understanding is Perfect!

```
START: V1 Propensity Models
  ↓ (built with historical data)
  ↓
A/B TEST (dual purpose)
  ├─ Validate V1 works
  └─ Collect randomized data
  ↓
END: V2 Uplift Models
  ↓ (built with experimental data)
  ↓
RESULT: Revenue with/without action predictions
```

**Uplift Model = Methodology using standard ML to predict causal effects**

Install these packages:
```bash
pip install causalml
pip install scikit-uplift
pip install econml
```

Then use **XGBoost/RandomForest/etc.** as base learners within the uplift framework!
