"""
Next Best Action: Complete Data Pipeline with Synthetic Example
Demonstration of: Historical Data → Initial Models → A/B Test → Uplift Models
Following FAANG standards (similar to Netflix, Uber, Amazon personalization)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# PHASE 1: HISTORICAL DATA (What We START With)
# ============================================================================
print("="*80)
print("PHASE 1: HISTORICAL DATA - Building Initial Models")
print("="*80)

# Simulate 10,000 customers with historical data
n_customers = 10000

historical_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'tenure_months': np.random.randint(1, 60, n_customers),
    'annual_revenue': np.random.choice([5000, 15000, 50000, 100000], n_customers, 
                                       p=[0.4, 0.3, 0.2, 0.1]),
    'engagement_score': np.random.randint(0, 100, n_customers),
    'support_tickets': np.random.poisson(2, n_customers),
    'feature_adoption': np.random.beta(2, 5, n_customers),  # Skewed towards lower
    'industry': np.random.choice(['Tech', 'Finance', 'Retail', 'Healthcare'], n_customers),
    'contract_length_months': np.random.choice([12, 24, 36], n_customers),
})

# Existing churn model produces churn probability (given from problem)
# Realistic relationship: low tenure, low engagement → high churn
historical_data['churn_probability'] = (
    0.7 * (1 - historical_data['tenure_months'] / 60) +
    0.3 * (1 - historical_data['engagement_score'] / 100) +
    np.random.normal(0, 0.1, n_customers)
).clip(0, 1)

# Historical actions taken (biased - we only called high-value customers!)
# THIS IS THE SELECTION BIAS PROBLEM
historical_data['action_taken'] = 'none'
historical_data.loc[
    (historical_data['churn_probability'] > 0.6) & 
    (historical_data['annual_revenue'] > 30000), 
    'action_taken'
] = 'retention_call'

historical_data.loc[
    (historical_data['churn_probability'] > 0.6) & 
    (historical_data['annual_revenue'] <= 30000), 
    'action_taken'
] = 'retention_email'

# Actual outcomes (observed)
# Ground truth: actions DO help, but we don't know counterfactual
def simulate_outcome(row):
    base_churn_prob = row['churn_probability']
    
    # True causal effects (UNKNOWN to us, only God/simulation knows)
    if row['action_taken'] == 'retention_call':
        base_churn_prob *= 0.6  # 40% reduction
    elif row['action_taken'] == 'retention_email':
        base_churn_prob *= 0.8  # 20% reduction
    
    return np.random.binomial(1, base_churn_prob)

historical_data['churned'] = historical_data.apply(simulate_outcome, axis=1)

print("\n Historical Data Sample:")
print(historical_data.head(10))

print("\n Historical Action Distribution (SELECTION BIAS!):")
print(historical_data['action_taken'].value_counts())

print("\n CRITICAL PROBLEM WITH HISTORICAL DATA:")
print(f"Churn rate for 'none' action: {historical_data[historical_data['action_taken']=='none']['churned'].mean():.1%}")
print(f"Churn rate for 'retention_call': {historical_data[historical_data['action_taken']=='retention_call']['churned'].mean():.1%}")
print("\n PARADOX: Customers with calls churn MORE! (selection bias - we called risky customers)")
print(" We CANNOT conclude calls don't work - we need randomized experiment!")

# ============================================================================
# PHASE 2: BUILD INITIAL MODELS (Using Historical Data)
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: INITIAL MODEL BUILDING")
print("="*80)

# Step 2.1: 3D Segmentation
def segment_customer(row):
    """Create 3D segments: Risk × CLV × Engagement"""
    # Risk tier
    if row['churn_probability'] > 0.6:
        risk = 'High'
    elif row['churn_probability'] > 0.3:
        risk = 'Medium'
    else:
        risk = 'Low'
    
    # Value tier (CLV simplified: revenue × (1-churn) × 2 years)
    clv = row['annual_revenue'] * (1 - row['churn_probability']) * 2
    if clv > 100000:
        value = 'High'
    elif clv > 40000:
        value = 'Medium'
    else:
        value = 'Low'
    
    # Engagement tier
    if row['engagement_score'] > 75:
        engagement = 'High'
    else:
        engagement = 'Low'
    
    return f"{risk}Risk_{value}Value_{engagement}Engagement"

historical_data['segment'] = historical_data.apply(segment_customer, axis=1)

print("\n Customer Segments:")
print(historical_data['segment'].value_counts().head(10))

# Step 2.2: Upsell Propensity Model
# Feature: Only for LOW risk customers based on historical upsell attempts
# Simulate some historical upsell attempts and outcomes
upsell_candidates = historical_data[historical_data['churn_probability'] < 0.3].copy()
upsell_candidates['upsell_attempted'] = np.random.binomial(1, 0.2, len(upsell_candidates))

# True upsell success depends on engagement and tenure
def simulate_upsell_success(row):
    if row['upsell_attempted'] == 0:
        return np.nan
    
    success_prob = (
        0.5 * (row['engagement_score'] / 100) +
        0.3 * (row['feature_adoption']) +
        0.2 * min(row['tenure_months'] / 36, 1)
    )
    return np.random.binomial(1, success_prob)

upsell_candidates['upsell_success'] = upsell_candidates.apply(simulate_upsell_success, axis=1)

# Train upsell propensity model on customers where we attempted upsell
upsell_train = upsell_candidates[upsell_candidates['upsell_attempted'] == 1].copy()

X_upsell = upsell_train[['engagement_score', 'feature_adoption', 'tenure_months', 'annual_revenue']]
y_upsell = upsell_train['upsell_success']

if len(y_upsell.dropna()) > 50:
    X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(
        X_upsell, y_upsell, test_size=0.3, random_state=42
    )
    
    upsell_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    upsell_model.fit(X_train_up, y_train_up)
    
    print("\n Upsell Propensity Model Performance:")
    print(f"AUC: {roc_auc_score(y_test_up, upsell_model.predict_proba(X_test_up)[:, 1]):.3f}")
    
    # Score all low-risk customers
    low_risk = historical_data[historical_data['churn_probability'] < 0.3].copy()
    X_score = low_risk[['engagement_score', 'feature_adoption', 'tenure_months', 'annual_revenue']]
    historical_data.loc[low_risk.index, 'upsell_propensity'] = upsell_model.predict_proba(X_score)[:, 1]
else:
    historical_data['upsell_propensity'] = 0
    print("\n Insufficient upsell training data - using rule-based approach")

# Step 2.3: Expected Value Framework & Action Assignment
def assign_nba_action(row):
    """NBA Decision Logic"""
    # Retention always wins for high risk
    if row['churn_probability'] >= 0.6:
        if row['annual_revenue'] >= 50000:
            return 'retention_call', 500  # Cost $500
        else:
            return 'retention_email', 50  # Cost $50
    
    # Upsell for low risk + high engagement
    elif row['churn_probability'] < 0.2 and row['engagement_score'] > 75:
        if row.get('upsell_propensity', 0) > 0.4:
            return 'upsell_offer', 100  # Cost $100
    
    # Monitor only - don't annoy
    return 'none', 0

historical_data[['nba_action', 'action_cost']] = historical_data.apply(
    lambda row: pd.Series(assign_nba_action(row)), axis=1
)

# Calculate Expected Value for each recommendation
def calculate_ev(row):
    if row['nba_action'] == 'retention_call':
        # EV = P(Save) × Revenue - Cost
        # Assumption: 40% success rate based on industry benchmarks
        return 0.4 * row['annual_revenue'] * 2 - row['action_cost']
    elif row['nba_action'] == 'retention_email':
        return 0.2 * row['annual_revenue'] * 2 - row['action_cost']
    elif row['nba_action'] == 'upsell_offer':
        # Assume upsell adds 30% revenue
        return row.get('upsell_propensity', 0) * row['annual_revenue'] * 0.3 - row['action_cost']
    return 0

historical_data['expected_value'] = historical_data.apply(calculate_ev, axis=1)

print("\n NBA Recommendations Summary:")
print(historical_data['nba_action'].value_counts())
print(f"\nTotal Expected Value: ${historical_data['expected_value'].sum():,.0f}")
print(f"Average EV per Action: ${historical_data[historical_data['nba_action']!='none']['expected_value'].mean():,.0f}")

# ============================================================================
# PHASE 3: A/B TEST SETUP (Data Collection for Validation)
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: A/B TEST - COLLECTING EXPERIMENTAL DATA")
print("="*80)

# Select customers in renewal window (next 90 days) for A/B test
renewal_customers = historical_data[
    (historical_data['tenure_months'] % 12 >= 9) &  # Within 3 months of renewal
    (historical_data['tenure_months'] % 12 < 12)
].copy()

print(f"\n Customers in Renewal Window: {len(renewal_customers)}")

# Stratified randomization: ensure balance across segments
from sklearn.model_selection import StratifiedShuffleSplit

stratifier = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
for control_idx, treatment_idx in stratifier.split(
    renewal_customers, 
    renewal_customers['segment']
):
    control_group = renewal_customers.iloc[control_idx].copy()
    treatment_group = renewal_customers.iloc[treatment_idx].copy()

control_group['ab_group'] = 'Control'
treatment_group['ab_group'] = 'Treatment'

print(f"\n A/B Test Groups:")
print(f"Control (Business as Usual): {len(control_group)} customers")
print(f"Treatment (NBA Recommendations): {len(treatment_group)} customers")

# Control: Business as usual (same biased logic as historical)
control_group['applied_action'] = 'none'
control_group.loc[
    (control_group['churn_probability'] > 0.6) & 
    (control_group['annual_revenue'] > 30000), 
    'applied_action'
] = 'retention_call'

# Treatment: Follow NBA recommendations
treatment_group['applied_action'] = treatment_group['nba_action']

# Simulate outcomes for 12 weeks
def simulate_ab_outcome(row):
    """Simulate outcome with TRUE causal effects"""
    base_churn_prob = row['churn_probability']
    upsell_revenue = 0
    
    # TRUE causal effects (what we're trying to discover!)
    if row['applied_action'] == 'retention_call':
        base_churn_prob *= 0.6  # 40% reduction (High touch)
    elif row['applied_action'] == 'retention_email':
        base_churn_prob *= 0.8  # 20% reduction (Automated)
    elif row['applied_action'] == 'upsell_offer':
        # Only works if truly engaged
        if row['engagement_score'] > 75:
            upsell_revenue = row['annual_revenue'] * 0.3 * row.get('upsell_propensity', 0.5)
    
    churned = np.random.binomial(1, base_churn_prob)
    
    # Revenue calculation
    if churned:
        revenue = 0
    else:
        revenue = row['annual_revenue'] + upsell_revenue
    
    return pd.Series({
        'churned': churned,
        'revenue': revenue,
        'action_cost': row.get('action_cost', 0)
    })

# Apply to both groups
ab_test_data = pd.concat([control_group, treatment_group], ignore_index=True)
ab_test_data[['churned', 'revenue', 'action_cost']] = ab_test_data.apply(
    simulate_ab_outcome, axis=1
)
ab_test_data['net_revenue'] = ab_test_data['revenue'] - ab_test_data['action_cost']

print("\n A/B Test Results:")
print("="*80)

# Primary metrics
for group in ['Control', 'Treatment']:
    group_data = ab_test_data[ab_test_data['ab_group'] == group]
    
    print(f"\n{group} Group:")
    print(f"  Retention Rate: {(1 - group_data['churned'].mean()) * 100:.1f}%")
    print(f"  Avg Revenue per Customer: ${group_data['revenue'].mean():,.0f}")
    print(f"  Avg Net Revenue (after costs): ${group_data['net_revenue'].mean():,.0f}")
    print(f"  Total Cost: ${group_data['action_cost'].sum():,.0f}")

# Calculate lift
control_retention = 1 - ab_test_data[ab_test_data['ab_group'] == 'Control']['churned'].mean()
treatment_retention = 1 - ab_test_data[ab_test_data['ab_group'] == 'Treatment']['churned'].mean()
retention_lift = (treatment_retention - control_retention) / control_retention * 100

control_revenue = ab_test_data[ab_test_data['ab_group'] == 'Control']['net_revenue'].mean()
treatment_revenue = ab_test_data[ab_test_data['ab_group'] == 'Treatment']['net_revenue'].mean()
revenue_lift = (treatment_revenue - control_revenue) / control_revenue * 100

print(f"\n INCREMENTAL IMPACT:")
print(f"  Retention Lift: +{retention_lift:.1f}%")
print(f"  Net Revenue Lift: +{revenue_lift:.1f}%")
print(f"  Annual Impact (extrapolated): ${(treatment_revenue - control_revenue) * len(renewal_customers) * 4:,.0f}")

# Statistical significance (simplified)
from scipy.stats import chi2_contingency, ttest_ind

# Chi-square for retention
contingency = pd.crosstab(
    ab_test_data['ab_group'], 
    ab_test_data['churned']
)
chi2, p_value_retention, _, _ = chi2_contingency(contingency)

# T-test for revenue
control_revenues = ab_test_data[ab_test_data['ab_group'] == 'Control']['net_revenue']
treatment_revenues = ab_test_data[ab_test_data['ab_group'] == 'Treatment']['net_revenue']
t_stat, p_value_revenue = ttest_ind(treatment_revenues, control_revenues)

print(f"\n Statistical Significance:")
print(f"  Retention p-value: {p_value_retention:.4f} {'✓ Significant' if p_value_retention < 0.05 else '✗ Not Significant'}")
print(f"  Revenue p-value: {p_value_revenue:.4f} {'✓ Significant' if p_value_revenue < 0.05 else '✗ Not Significant'}")

# ============================================================================
# PHASE 4: POST-A/B - UPLIFT MODELING (Using Experimental Data)
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: UPLIFT MODELING - Learning from A/B Test")
print("="*80)

print("\n🎓 What We Learned from A/B Test:")
print("1. We now have CAUSAL data - know what WOULD have happened without action")
print("2. Can identify TRUE heterogeneous treatment effects")
print("3. Build uplift models to predict: Uplift = Outcome(Treatment) - Outcome(Control)")

# Simple uplift approach: Two-model method
# Model 1: P(Positive Outcome | Treatment=1)
# Model 2: P(Positive Outcome | Treatment=0)
# Uplift = Model1 - Model2

treatment_data = ab_test_data[ab_test_data['ab_group'] == 'Treatment'].copy()
control_data = ab_test_data[ab_test_data['ab_group'] == 'Control'].copy()

# For retention actions
retention_treatment = treatment_data[treatment_data['applied_action'].isin(['retention_call', 'retention_email'])].copy()
retention_control = control_data.copy()

if len(retention_treatment) > 50 and len(retention_control) > 50:
    X_cols = ['churn_probability', 'annual_revenue', 'engagement_score', 'tenure_months']
    
    # Model for treatment group
    X_treat = retention_treatment[X_cols]
    y_treat = 1 - retention_treatment['churned']  # 1 = retained
    
    # Model for control group
    X_ctrl = retention_control[X_cols]
    y_ctrl = 1 - retention_control['churned']
    
    uplift_model_treat = GradientBoostingClassifier(n_estimators=50, random_state=42)
    uplift_model_control = GradientBoostingClassifier(n_estimators=50, random_state=42)
    
    uplift_model_treat.fit(X_treat, y_treat)
    uplift_model_control.fit(X_ctrl, y_ctrl)
    
    print("\n Uplift Models Trained!")
    
    # Predict uplift for a sample customer
    sample_customer = pd.DataFrame({
        'churn_probability': [0.7],
        'annual_revenue': [75000],
        'engagement_score': [60],
        'tenure_months': [24]
    })
    
    p_treat = uplift_model_treat.predict_proba(sample_customer)[0, 1]
    p_control = uplift_model_control.predict_proba(sample_customer)[0, 1]
    uplift = p_treat - p_control
    
    print(f"\n Example Customer Prediction:")
    print(f"  Retention Prob (With Action): {p_treat:.1%}")
    print(f"  Retention Prob (Without Action): {p_control:.1%}")
    print(f"   UPLIFT (Incremental Benefit): {uplift:.1%}")
    print(f"  → Decision: {'✓ Intervene (positive uplift)' if uplift > 0.05 else '✗ Skip (low uplift)'}")

# ============================================================================
# KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" KEY INSIGHTS: The Data Journey")
print("="*80)

print("""
PHASE 1 - HISTORICAL DATA (Start Here):
├─ What We Have: Past customer behavior, outcomes, biased actions
├─ Problem: Selection bias - don't know counterfactuals
└─ Use Case: Build INITIAL propensity models, segmentation

PHASE 2 - INITIAL MODELS (Before A/B):
├─ Build: Upsell propensity, segments, EV framework
├─ Output: NBA recommendations (educated guesses)
└─ Gap: Don't know if they ACTUALLY work better than current approach

PHASE 3 - A/B TEST (Dual Purpose):
├─ Purpose 1: VALIDATE initial models work (are they better than status quo?)
├─ Purpose 2: COLLECT experimental data (randomized outcomes)
├─ Duration: 12 weeks (need statistical power)
└─ Output: Causal data showing true incremental impact

PHASE 4 - UPLIFT MODELS (After A/B):
├─ Input: Experimental data from A/B test
├─ Build: True uplift models (heterogeneous treatment effects)
├─ Predict: Who benefits MOST from intervention (not just likely to respond)
└─ Redeploy: Much more precise targeting

TIMELINE:
Week 1-8:   Build initial models using historical data
Week 9-20:  Run A/B test (validate + collect data)
Week 21-26: Build uplift models, redeploy improved system
Ongoing:    Continuous learning loop
""")

print("\n Example Data Saved:")
ab_test_data.to_csv('/Users/marghubakhtar/Documents/Recommendation_Systems_Production/ab_test_results.csv', index=False)
print("✓ ab_test_results.csv - Full A/B test results with outcomes")

historical_data.to_csv('/Users/marghubakhtar/Documents/Recommendation_Systems_Production/historical_data_with_nba.csv', index=False)
print("✓ historical_data_with_nba.csv - Historical data with NBA recommendations")

print("\n" + "="*80)
print(" Pipeline Complete - Ready for Production!")
print("="*80)
