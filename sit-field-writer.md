Gather what the program does, who uses its output, and what they do with it specifically from the user. If you don't have those stop and ask. Then walk the construction graph from leaves to roots. For each field, write a `Field(description=...)` that:

1. Uses the fewest tokens that leave zero ambiguity about what the value means in this program's domain
2. Says only what the type annotation doesn't already say — the type handled its part, the description handles the rest
3. Grounds in what the value tells the consumer about the thing being analyzed, not how the program computed it

Use the vocabulary of the domain (type analysis, field classification, model structure) not the vocabulary of the implementation (validators, construction phases, internal wiring).

Examples:

- `credit_limit: Decimal` → `"Maximum outstanding balance the borrower is approved for, in USD"`
- `churn_risk: RiskTier` → `"Likelihood of voluntary departure within 90 days — CRITICAL, HIGH, MODERATE, or LOW"`
- `auto_renew: bool` → `"True when the subscription renews automatically at period end without customer action"`

Do not explain the codebase. Do not teach concepts. Do not reference frameworks or architectural patterns. Each description is an instruction that resolves what this field's value means for the consumer holding the result.
