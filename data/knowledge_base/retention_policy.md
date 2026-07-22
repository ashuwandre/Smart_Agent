# Retention Offer Policy

**Policy version:** 1.1  
**Effective date:** 2026-07-01

## Eligibility

A retention discount may be considered only when all conditions are true:

1. The customer clearly indicates an intention to cancel or leave because of price or value.
2. The account is active and has no outstanding invoice.
3. Tenure is at least 12 months.
4. The churn risk score is 0.70 or higher.
5. No retention offer was already granted to the customer in the current run.

The agent must verify eligibility using customer data; it must not infer missing values.

## Discount caps

- Every plan has a maximum retention discount of 20% for 3 months.

The offered percentage may be lower than the cap. It must never exceed 20%.

## Run budget

The total cost of all autonomous retention offers in one run is capped at ₹10,000. Offer cost is calculated as `monthly_fee × discount_percentage × 3 months`. If the remaining budget is insufficient, do not grant the offer; escalate for human review. Offers must not be split or duplicated to bypass the budget.

Customers who are ineligible may still cancel normally or be escalated if they dispute eligibility. Do not describe a discount as guaranteed until the grant tool confirms success.
