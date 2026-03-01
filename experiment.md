# Experiment Design: Semantic Index Types

## File Structure

```
compose.yaml                  ollama + GPU for local Granite runs

sit/
  __init__.py
  experiment.py               runner: API call loop, resumable
  scoring.py                  accuracy, JS divergence, ΔH_f
  schemas/
    __init__.py
    baseline.py               correct semantic indices (TCA-quality)
    vacuous.py                absent semantic indices (field_1, OPTION_A)
    misleading.py             wrong-domain semantic indices (discount tiers)
  prompts/
    __init__.py
    profiles.json             50 customer profiles + ground truth labels
    README.md                 data source and derivation docs

Models: Claude Sonnet 4.6 (API), ChatGPT 5.2 (API), Granite 19B (local/ollama)
```

## Goal

Directly instantiate the core claim (§2.3) in the paper's target domain: Pydantic structured output consumed by language models. Demonstrate that structurally isomorphic schemas with different natural-language content produce measurably different output distributions while maintaining 100% structural validity.

## Schema

One Pydantic model: a customer retention risk analysis. The model includes:

- Enum fields with clear categorical semantics (risk tier, engagement level, churn driver)
- A list field for recommended interventions
- A confidence score
- A text field for reasoning/summary

The model is designed so that enum fields have unambiguous correct answers given a well-constructed prompt, enabling direct accuracy measurement.

## Three Structurally Isomorphic Variants

All variants share identical structure: same field count, same types, same enum cardinality, same nesting depth. They differ only in natural-language content (field names, descriptions, enum member names).

**Baseline.** Domain-precise names and descriptions. `churn_risk_tier: ChurnRiskTier` with description "Likelihood of voluntary departure within 90 days based on behavioral signals." Enum members: `CRITICAL`, `HIGH`, `MODERATE`, `LOW`. This is the schema as a practitioner would write it.

**Vacuous.** Structural identifiers only. `field_1: Category4` with description "" or "A category value." Enum members: `OPTION_A`, `OPTION_B`, `OPTION_C`, `OPTION_D`. Same structure, no semantic content. Tests whether removing the semantic channel degrades output quality while structural validity holds.

**Misleading.** Wrong-domain names, same structure. `discount_offer_level: DiscountOfferLevel` with description "The promotional discount tier to extend to the customer." Enum members: `PLATINUM`, `GOLD`, `SILVER`, `BRONZE`. Tests whether the model computes a *different thing* when the semantic index points elsewhere — not wrong output, but divergent output.

## Prompts

50 customer profiles derived from the public Telco Customer Churn dataset (`scikit-learn/churn-prediction` on Hugging Face, 7,040 rows). Using a public dataset instead of fully synthetic prompts gives reproducibility, realistic feature distributions, and defensibility — we didn't cherry-pick scenarios to make results look good.

### Derivation process

1. Sample 50 customers from the dataset with a spread across feature space (tenure, monthly charges, contract type, support tickets, services used, payment method).
2. Convert each customer's feature vector into a natural-language paragraph. Example: "This customer has been with us for 2 months on a month-to-month contract, paying $89/month. They called support 4 times last quarter and have no additional services beyond basic phone."
3. Hand-label a 4-level risk tier (CRITICAL/HIGH/MODERATE/LOW) for each customer based on the features. The source dataset has binary churn/no-churn labels; we derive 4 tiers from signal strength (e.g., churned + short tenure + high charges + many support calls = CRITICAL; not churned + long tenure + annual contract = LOW). This derivation is a judgment call but is documented and reproducible.
4. Distribute across the risk spectrum: ~12 CRITICAL, ~13 HIGH, ~13 MODERATE, ~12 LOW.

Each profile is a plain text paragraph. No schema-specific language appears in the prompt — the prompt describes a customer, not a risk assessment. The schema is the only source of analytical framing.

## Models

- Claude Sonnet 4.6 (Anthropic) — via tool use structured output
- ChatGPT 5.2 (OpenAI) — via structured outputs with `strict: true`

Both enforce structural validity through their respective constrained generation mechanisms. Two models establish that the phenomenon is a property of neural consumers generally, not a vendor-specific artifact.

## Sampling

- Temperature: 1.0
- Samples per (prompt, variant, model): 20
- Total API calls: 50 prompts × 3 variants × 2 models × 20 samples = 6,000

Temperature 1.0 with 20 samples per condition produces distributions stable enough to compute divergence metrics while capturing the full range of the model's output variation.

## Measurements

### Primary: Enum Field Accuracy

Each enum field has a hand-labeled ground truth per prompt. Accuracy is computed as the fraction of samples matching the ground truth label, reported per variant per model.

Expected pattern:
- Baseline: high accuracy (model has precise semantic guidance)
- Vacuous: degraded accuracy (model has no semantic guidance, selects within valid set without interpretive grounding)
- Misleading: divergent from baseline (model computes a different analytical frame — not random, but systematically different)

### Secondary: Distribution Divergence

Jensen-Shannon divergence between the output distributions of each variant pair (baseline vs vacuous, baseline vs misleading) for each enum field. This quantifies how much the semantic channel shifts the output distribution within the structurally valid space.

### Structural Validity

100% of outputs across all variants and models must be structurally valid. This is guaranteed by constrained decoding but is reported explicitly to demonstrate the structural channel's hard guarantee.

### Qualitative: Text Field Examples

Cherry-picked examples from the text/list fields (reasoning, recommended interventions) showing semantic drift across variants. These are not scored quantitatively — they illustrate what changes when the semantic index changes, for the reader who wants to see it rather than measure it.

## What This Demonstrates

1. **Structural channel holds universally.** Every output across every variant is schema-valid. The structural constraint is invariant under renaming.
2. **Semantic channel is not invariant under renaming.** Accuracy and output distributions shift measurably across variants. The same structural space produces different semantic content depending on the natural-language indices.
3. **The phenomenon is cross-model.** Both Claude Sonnet 4.6 and ChatGPT 5.2 exhibit the pattern, establishing it as a property of neural consumers rather than a model-specific behavior.
4. **This is §2.3 instantiated.** Structurally isomorphic schemas ($S \equiv_\alpha S'$) that produce measurably different output distributions ($S \not\equiv_{\text{sem}} S'$), demonstrated in the exact host system the paper describes.
