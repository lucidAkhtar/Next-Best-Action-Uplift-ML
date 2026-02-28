
## 5. Illustrative Code: NBA Engine Core Logic

```python
import pandas as pd

class NBAEngine:
    def __init__(self, upsell_model, treatment_model):
        self.upsell_model = upsell_model
        self.treatment_model = treatment_model
    
    def calculate_clv(self, customer):
        return customer['annual_revenue'] * (1 - customer['churn_probability']) * 3
    
    def segment_customer(self, customer):
        risk, clv, eng = customer['churn_probability'], self.calculate_clv(customer), customer['engagement_score']
        if risk > 0.6 and clv > 50000: return "critical_save"
        elif risk > 0.6: return "automated_retention"
        elif risk < 0.2 and clv > 50000 and eng > 75: return "upsell_opportunity"
        elif risk < 0.3 and clv > 50000: return "monitor_only"
        return "standard_service"
    
    def calculate_ev(self, customer, action):
        clv = self.calculate_clv(customer)
        # Success rates from industry benchmarks: personal outreach 60-75%, automated 30-45%
        if action == "account_call": return 0.7 * clv - 500  # 70% historical response
        elif action == "discount_email": return 0.4 * (clv * 0.8) - (customer['annual_revenue'] * 0.2)  # 40% response, 20% discount
        elif action == "upsell": return self.upsell_model.predict_proba([customer])[0][1] * 10000 - 200  # Model-based
        return 0
    
    def recommend_action(self, customer):
        segment = self.segment_customer(customer)
        actions = {"critical_save": ["account_call", "discount_email"],
                   "automated_retention": ["discount_email"],
                   "upsell_opportunity": ["upsell"]}.get(segment, [])
        
        if not actions: return {"action": "no_action", "ev": 0, "segment": segment}
        
        evs = {a: self.calculate_ev(customer, a) for a in actions}
        best = max(evs, key=evs.get)
        return {"action": best, "ev": evs[best], "segment": segment, "customer_id": customer['customer_id']}
    
    def batch_recommendations(self, customer_df):
        recs = [self.recommend_action(row) for _, row in customer_df.iterrows()]
        rec_df = pd.DataFrame(recs).sort_values('ev', ascending=False)
        
        # Apply constraints: $100K budget, 50 calls/week capacity
        call_mask = rec_df['action'] == 'account_call'
        rec_df.loc[call_mask & (rec_df.index >= 50), 'action'] = 'discount_email'  # Downgrade beyond capacity
        return rec_df.head(500)  # Top 500 by EV

# Usage: daily_recs = NBAEngine(upsell_model, treatment_model).batch_recommendations(renewal_customers)
```