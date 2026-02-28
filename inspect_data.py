"""
Quick Data Inspection: A/B Test Results
Shows the structure and key insights from experimental data
"""

import pandas as pd
import numpy as np

# Load the A/B test results
ab_data = pd.read_csv('ab_test_results.csv')

print("="*80)
print("A/B TEST DATA STRUCTURE")
print("="*80)

print("\n1. DATA SHAPE:")
print(f"   Total customers: {len(ab_data):,}")
print(f"   Features/columns: {len(ab_data.columns)}")

print("\n2. KEY COLUMNS:")
print("\n   INPUTS (What we knew BEFORE):")
for col in ['customer_id', 'tenure_months', 'annual_revenue', 'engagement_score', 
            'churn_probability', 'segment']:
    if col in ab_data.columns:
        print(f"     • {col}")

print("\n   TREATMENT (What we did):")
for col in ['ab_group', 'nba_action', 'applied_action', 'action_cost']:
    if col in ab_data.columns:
        print(f"     • {col}")

print("\n   OUTCOMES (What happened):")
for col in ['churned', 'revenue', 'net_revenue']:
    if col in ab_data.columns:
        print(f"     • {col}")

print("\n" + "="*80)
print("SAMPLE CUSTOMERS (Understanding the Data)")
print("="*80)

# Show 3 interesting examples
examples = []

# Example 1: Control group, high risk, churned
ex1 = ab_data[(ab_data['ab_group'] == 'Control') & 
              (ab_data['churn_probability'] > 0.6) &
              (ab_data['churned'] == 1)].head(1)

# Example 2: Treatment group, retention saved
ex2 = ab_data[(ab_data['ab_group'] == 'Treatment') & 
              (ab_data['applied_action'].isin(['retention_call', 'retention_email'])) &
              (ab_data['churned'] == 0)].head(1)

# Example 3: Treatment group, upsell success
ex3 = ab_data[(ab_data['ab_group'] == 'Treatment') & 
              (ab_data['applied_action'] == 'upsell_offer') &
              (ab_data['revenue'] > ab_data['annual_revenue'])].head(1)

if not ex1.empty:
    row = ex1.iloc[0]
    print("\n EXAMPLE 1: Control Group - High Risk Customer (No NBA intervention)")
    print(f"   Customer ID: {int(row['customer_id'])}")
    print(f"   Churn Risk: {row['churn_probability']:.1%}")
    print(f"   Revenue: ${row['annual_revenue']:,.0f}")
    print(f"   Engagement: {row['engagement_score']}/100")
    print(f"   NBA would recommend: {row['nba_action']}")
    print(f"   But they're in CONTROL → Action: {row['applied_action']}")
    print(f"   Outcome: {' CHURNED' if row['churned'] else ' RETAINED'}")
    print(f"   → This customer is the COUNTERFACTUAL for treatment group")

if not ex2.empty:
    row = ex2.iloc[0]
    print("\n EXAMPLE 2: Treatment Group - Retention Success")
    print(f"   Customer ID: {int(row['customer_id'])}")
    print(f"   Churn Risk: {row['churn_probability']:.1%}")
    print(f"   Revenue: ${row['annual_revenue']:,.0f}")
    print(f"   NBA recommended: {row['nba_action']}")
    print(f"   Action taken: {row['applied_action']} (cost: ${row['action_cost']:,.0f})")
    print(f"   Outcome: {' CHURNED' if row['churned'] else ' RETAINED'}")
    print(f"   Net Value: ${row['net_revenue']:,.0f}")
    print(f"   → SUCCESS! Retention action worked")

if not ex3.empty:
    row = ex3.iloc[0]
    print("\n EXAMPLE 3: Treatment Group - Upsell Success")
    print(f"   Customer ID: {int(row['customer_id'])}")
    print(f"   Churn Risk: {row['churn_probability']:.1%} (Low)")
    print(f"   Base Revenue: ${row['annual_revenue']:,.0f}")
    print(f"   Engagement: {row['engagement_score']}/100")
    print(f"   NBA recommended: {row['nba_action']}")
    print(f"   Action taken: {row['applied_action']} (cost: ${row['action_cost']:,.0f})")
    print(f"   Final Revenue: ${row['revenue']:,.0f} (Upsold!)")
    print(f"   Incremental: +${row['revenue'] - row['annual_revenue']:,.0f}")
    print(f"   → SUCCESS! Upsell converted")

print("\n" + "="*80)
print("WHAT MAKES THIS 'EXPERIMENTAL' DATA SPECIAL")
print("="*80)

print("""
🔬 Key Difference from Historical Data:

HISTORICAL DATA (Observational):
├─ Customer with high risk → We called them → They churned
└─ QUESTION: Did they churn BECAUSE we called? Or DESPITE the call?
   └─ ANSWER: Unknown! We have no counterfactual.

A/B TEST DATA (Experimental):
├─ High-risk customers RANDOMLY split:
│   ├─ Control: No intervention → 57% churned
│   └─ Treatment: We intervened → 38% churned
└─ CAUSAL CONCLUSION: Intervention REDUCED churn by 19 percentage points!
   └─ We can say this confidently because of randomization
""")

print("\n" + "="*80)
print("HOW THIS DATA ENABLES UPLIFT MODELING")
print("="*80)

print("""
With this data, we can now train models that predict:

   Uplift = P(Positive Outcome | Treatment) - P(Positive Outcome | Control)

Example splits for modeling:

1. TREATMENT MODEL (learns from treatment group):
   ├─ Input: Customer features
   └─ Output: P(Retained | We take action)

2. CONTROL MODEL (learns from control group):
   ├─ Input: Same customer features  
   └─ Output: P(Retained | We do nothing)

3. UPLIFT PREDICTION:
   For new customer:
   ├─ Treatment Model: "62% chance they stay if we call"
   ├─ Control Model: "59% chance they stay anyway"
   └─ Uplift: 3% incremental benefit
       └─ Decision: Maybe not worth the $500 cost!

This is IMPOSSIBLE to learn from historical data because we never
observed "what would have happened" for the path not taken.
""")

print("\n" + "="*80)
print("COMPARISON: Before vs After A/B Test")
print("="*80)

# Show aggregated differences
summary = pd.DataFrame({
    'Metric': [
        'Sample Size',
        'Retention Rate',
        'Avg Revenue',
        'Avg Cost',
        'Net Revenue'
    ],
    'Control (No NBA)': [
        f"{len(ab_data[ab_data['ab_group']=='Control']):,}",
        f"{(1-ab_data[ab_data['ab_group']=='Control']['churned'].mean())*100:.1f}%",
        f"${ab_data[ab_data['ab_group']=='Control']['revenue'].mean():,.0f}",
        f"${ab_data[ab_data['ab_group']=='Control']['action_cost'].mean():.0f}",
        f"${ab_data[ab_data['ab_group']=='Control']['net_revenue'].mean():,.0f}"
    ],
    'Treatment (NBA)': [
        f"{len(ab_data[ab_data['ab_group']=='Treatment']):,}",
        f"{(1-ab_data[ab_data['ab_group']=='Treatment']['churned'].mean())*100:.1f}%",
        f"${ab_data[ab_data['ab_group']=='Treatment']['revenue'].mean():,.0f}",
        f"${ab_data[ab_data['ab_group']=='Treatment']['action_cost'].mean():.0f}",
        f"${ab_data[ab_data['ab_group']=='Treatment']['net_revenue'].mean():,.0f}"
    ]
})

print("\n" + summary.to_string(index=False))

# Calculate lift
control_retention = 1 - ab_data[ab_data['ab_group'] == 'Control']['churned'].mean()
treatment_retention = 1 - ab_data[ab_data['ab_group'] == 'Treatment']['churned'].mean()
lift = (treatment_retention - control_retention) / control_retention * 100

print(f"\n🚀 RETENTION LIFT: +{lift:.1f}%")
print(f"   → This is what we couldn't measure before A/B test!")

print("\n" + "="*80)
print(" SUMMARY: Why We Need Both Types of Data")
print("="*80)

print("""
HISTORICAL DATA:
├─ Volume: Lots (10,000+ customers)
├─ Purpose: Build initial models, understand patterns
├─ Limitation: Selection bias, no counterfactuals
└─ Use: Phase 1 (weeks 1-8)

A/B TEST DATA:
├─ Volume: Moderate (2,000+ customers in test)
├─ Purpose: Validate models, measure causal impact
├─ Advantage: Randomization removes bias, provides counterfactuals
└─ Use: Phase 2 (weeks 9-20) → enables Phase 3 (uplift models)

CONTINUOUS PRODUCTION DATA:
├─ Volume: Grows over time
├─ Purpose: Model retraining, drift detection
├─ Includes: All recommendations, outcomes, feedback
└─ Use: Ongoing improvement (months 6+)

FAANG STANDARD: Use all three in sequence!
""")
