# NBA Engine - Illustrative Code
**Next Best Action System for Customer Retention & Upselling**  
**Candidate:** Md Marghub Akhtar 

---

## Purpose

This code illustrates how the analytical framework translates into action assignment logic. The focus is on clarity and structure rather than production implementation depth.

---

## NBA Engine Core Logic

```python
import pandas as pd

class NBAEngine:
    """
    Next Best Action recommendation engine combining segmentation,
    expected value calculation, and constraint-based optimization.
    """
    
    def __init__(self, upsell_model, treatment_model):
        """
        Initialize NBA engine with trained predictive models.
        
        Args:
            upsell_model: Trained model for upsell propensity prediction
            treatment_model: Trained model for treatment response prediction
        """
        self.upsell_model = upsell_model
        self.treatment_model = treatment_model
    
    def calculate_clv(self, customer):
        """
        Calculate Customer Lifetime Value.
        
        Formula: Annual Revenue × (1 - Churn Probability) × Expected Tenure
        
        Args:
            customer: Dict/Series with 'annual_revenue' and 'churn_probability'
            
        Returns:
            float: Estimated CLV over 3-year horizon
        """
        return customer['annual_revenue'] * (1 - customer['churn_probability']) * 3
    
    def segment_customer(self, customer):
        """
        Assign customer to analytical segment based on risk, value, and engagement.
        
        Segmentation Logic:
        - Critical Save: High risk (>60%) + High value (>$50K CLV)
        - Automated Retention: High risk + Low value
        - Upsell Opportunity: Low risk (<20%) + High value + High engagement (>75)
        - Monitor Only: Low risk + High value + Low engagement
        - Standard Service: All others
        
        Args:
            customer: Dict/Series with customer attributes
            
        Returns:
            str: Segment name
        """
        risk = customer['churn_probability']
        clv = self.calculate_clv(customer)
        engagement = customer['engagement_score']
        
        if risk > 0.6 and clv > 50000:
            return "critical_save"
        elif risk > 0.6:
            return "automated_retention"
        elif risk < 0.2 and clv > 50000 and engagement > 75:
            return "upsell_opportunity"
        elif risk < 0.3 and clv > 50000:
            return "monitor_only"
        else:
            return "standard_service"
    
    def calculate_ev(self, customer, action):
        """
        Calculate Expected Value for a customer-action pair.
        
        Formula: EV = P(Success) × Incremental_Value - Cost_of_Action
        
        Success rates from industry benchmarks:
        - Personal outreach (account manager calls): 60-75%
        - Automated outreach (emails): 30-45%
        
        Args:
            customer: Dict/Series with customer attributes
            action: Action type ('account_call', 'discount_email', 'upsell')
            
        Returns:
            float: Expected value in dollars
        """
        clv = self.calculate_clv(customer)
        
        if action == "account_call":
            # 70% historical response rate for personal outreach
            # $500 cost (account manager time)
            success_prob = 0.7
            incremental_value = clv  # Retain full customer value
            cost = 500
            return success_prob * incremental_value - cost
            
        elif action == "discount_email":
            # 40% response rate for automated discount offers
            # 20% discount reduces effective CLV
            success_prob = 0.4
            incremental_value = clv * 0.8  # Account for 20% discount
            cost = customer['annual_revenue'] * 0.2  # Discount cost
            return success_prob * incremental_value - cost
            
        elif action == "upsell":
            # Use predictive model for upsell success probability
            # $10K average upsell value, $200 outreach cost
            success_prob = self.upsell_model.predict_proba([customer])[0][1]
            incremental_value = 10000
            cost = 200
            return success_prob * incremental_value - cost
            
        else:  # no_action
            return 0
    
    def recommend_action(self, customer):
        """
        Recommend optimal action for a single customer.
        
        Logic:
        1. Segment customer
        2. Identify candidate actions for segment
        3. Calculate EV for each action
        4. Select action with highest EV
        
        Args:
            customer: Dict/Series with customer attributes
            
        Returns:
            dict: Recommendation with action, EV, segment, customer_id
        """
        segment = self.segment_customer(customer)
        
        # Map segments to candidate actions
        action_map = {
            "critical_save": ["account_call", "discount_email"],
            "automated_retention": ["discount_email"],
            "upsell_opportunity": ["upsell"]
        }
        
        candidate_actions = action_map.get(segment, [])
        
        # No action needed for monitor_only and standard_service segments
        if not candidate_actions:
            return {
                "action": "no_action",
                "expected_value": 0,
                "segment": segment,
                "customer_id": customer['customer_id']
            }
        
        # Calculate EV for each candidate action
        action_evs = {
            action: self.calculate_ev(customer, action) 
            for action in candidate_actions
        }
        
        # Select action with highest EV
        best_action = max(action_evs, key=action_evs.get)
        ev = action_evs[best_action]
        
        return {
            "action": best_action,
            "expected_value": ev,
            "priority": "high" if ev > 5000 else "medium",
            "segment": segment,
            "customer_id": customer['customer_id']
        }
    
    def batch_recommendations(self, customer_df):
        """
        Generate recommendations for all customers in renewal window.
        
        Process:
        1. Generate recommendations for each customer
        2. Rank by expected value
        3. Apply budget and capacity constraints
        4. Return top N prioritized recommendations
        
        Args:
            customer_df: DataFrame with customer attributes
            
        Returns:
            DataFrame: Recommendations ranked by expected value
        """
        recommendations = []
        
        # Generate recommendation for each customer
        for _, customer in customer_df.iterrows():
            rec = self.recommend_action(customer)
            recommendations.append(rec)
        
        # Convert to DataFrame and sort by expected value
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('expected_value', ascending=False)
        
        # Apply budget and capacity constraints
        rec_df = self._apply_budget_constraints(rec_df)
        
        return rec_df
    
    def _apply_budget_constraints(self, rec_df):
        """
        Apply operational constraints to recommendations.
        
        Constraints:
        - Budget: Max $100K/month for retention discounts
        - Capacity: Max 50 account manager calls per week
        
        Strategy: Prioritize highest EV recommendations within constraints.
        If capacity exceeded, downgrade action to next-best alternative.
        
        Args:
            rec_df: DataFrame with recommendations
            
        Returns:
            DataFrame: Recommendations adjusted for constraints
        """
        # Constraint: Max 50 account manager calls per week
        call_capacity = 50
        
        call_mask = rec_df['action'] == 'account_call'
        call_actions = rec_df[call_mask]
        
        # If exceeding capacity, downgrade lowest-EV calls to email
        if len(call_actions) > call_capacity:
            # Keep top 50 calls, downgrade rest to discount_email
            excess_indices = call_actions.index[call_capacity:]
            rec_df.loc[excess_indices, 'action'] = 'discount_email'
            rec_df.loc[excess_indices, 'expected_value'] *= 0.6  # Adjust EV for downgraded action
        
        # Return top 500 recommendations by expected value
        # (Assumption: Sales team can handle ~500 actions per week)
        return rec_df.head(500)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Assume models are pre-trained
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Mock models (in production, these would be trained models)
    upsell_model = GradientBoostingClassifier()  # Pre-trained
    treatment_model = GradientBoostingClassifier()  # Pre-trained
    
    # Initialize NBA engine
    nba_engine = NBAEngine(upsell_model, treatment_model)
    
    # Load customers approaching renewal (30-90 days out)
    renewal_customers = pd.read_csv('renewal_customers.csv')  # Example data source
    
    # Generate daily recommendations
    daily_recommendations = nba_engine.batch_recommendations(renewal_customers)
    
    # Export to CRM
    daily_recommendations.to_csv('daily_nba_recommendations.csv', index=False)
    
    # Example output:
    # customer_id | action           | expected_value | priority | segment
    # ------------|------------------|----------------|----------|------------------
    # C12345      | account_call     | 28500          | high     | critical_save
    # C67890      | upsell           | 7200           | high     | upsell_opportunity
    # C11223      | discount_email   | 4100           | medium   | automated_retention
    # C44556      | no_action        | 0              | low      | monitor_only
```

---

## Key Design Principles Illustrated

1. **Segmentation-First:** Customers grouped before action assignment (interpretable, explainable)
2. **Expected Value:** Every action justified by economics, not just propensity
3. **Constraint-Aware:** Capacity limits enforced automatically (realistic deployment)
4. **Modular:** Easy to adjust thresholds, add new segments, modify EV calculations
5. **Production-Ready Structure:** Class-based, well-documented, testable

---

## Model Integration Points

**This code assumes:**
- `upsell_model` and `treatment_model` are pre-trained scikit-learn compatible models
- Input features: `churn_probability`, `annual_revenue`, `engagement_score`, `customer_id`
- Models support `.predict_proba()` for probability outputs

**In production, these models would be:**
- Trained using XGBoost/LightGBM on historical customer data
- Versioned in MLflow model registry
- Containerized and served via REST API or batch inference

---

## Assumptions & Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CLV Horizon | 3 years | Gartner B2B benchmark (2.5-4 years) |
| High Value Threshold | $50K CLV | Top quartile assumption (would calibrate with data) |
| High Risk Threshold | 60% churn prob | Tertile split for subscription businesses |
| Engagement Threshold | 75 | Top quartile of engagement distribution |
| Account Call Success | 70% | Industry benchmark for personal outreach |
| Email Success | 40% | Industry benchmark for automated campaigns |
| Call Capacity | 50/week | Operational constraint (sales team size) |

**These would be validated and tuned during exploration phase with actual data.**
