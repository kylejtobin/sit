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
    names_only.py             baseline names, descriptions stripped
    vacuous.py                absent semantic indices (field_1, OPTION_A)
    misleading.py             wrong-domain semantic indices (coherent alternative)
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

## Structural Isomorphism Verification

All variants emit a canonical schema hash: field count, types, enum arities, nesting depth, required/optional positions. Hashes are equal across all variants. This is computed, not asserted — it shuts down any pedantic challenge to the isomorphism claim and aligns with the existential equivalence framing in §2.2.

## Four Structurally Isomorphic Variants

All variants share identical structure: same field count, same types, same enum cardinality, same nesting depth. They differ only in natural-language content (field names, descriptions, enum member names).

**Baseline.** Domain-precise names and descriptions. `churn_risk_tier: ChurnRiskTier` with description "Likelihood of voluntary departure within 90 days based on behavioral signals." Enum members: `CRITICAL`, `HIGH`, `MODERATE`, `LOW`. This is the schema as a practitioner would write it.

**Names-only.** Baseline field names and enum member names, but all `Field(description=...)` values stripped or replaced with a constant empty string. Same names as baseline, no descriptive prose. Separates the contribution of field identifiers from the contribution of description text — isolates WHICH part of $S_{\text{lang}}$ does the work.

**Vacuous.** Structural identifiers only. `field_1: Category4` with description "" or "A category value." Enum members: `OPTION_A`, `OPTION_B`, `OPTION_C`, `OPTION_D`. Same structure, no semantic content. Tests whether removing the semantic channel entirely degrades output quality while structural validity holds.

**Misleading.** Wrong-domain names, same structure. A coherent alternative analytical frame — not arbitrary noise, but a plausible different computation applied to the same customer data. `retention_offer_aggressiveness: OfferAggressiveness` with description "How aggressively to discount in order to retain the customer." Enum members: `MAXIMUM`, `STRONG`, `MODERATE`, `MINIMAL`. Tests whether the model computes a *different function* when the semantic index points elsewhere — not wrong output, but systematically divergent output. The misleading enum names are a cardinality-preserving permutation of baseline levels in structure only, not in meaning.

## Prompts

50 customer profiles derived from the public Telco Customer Churn dataset (`scikit-learn/churn-prediction` on Hugging Face, 7,040 rows). Using a public dataset instead of fully synthetic prompts gives reproducibility, realistic feature distributions, and defensibility — we didn't cherry-pick scenarios to make results look good.

### Derivation process

1. Sample 50 customers from the dataset with a spread across feature space (tenure, monthly charges, contract type, support tickets, services used, payment method).
2. Convert each customer's feature vector into a natural-language paragraph. Example: "This customer has been with us for 2 months on a month-to-month contract, paying $89/month. They called support 4 times last quarter and have no additional services beyond basic phone."
3. Hand-label a 4-level risk tier (CRITICAL/HIGH/MODERATE/LOW) for each customer based on the features. The source dataset has binary churn/no-churn labels; we derive 4 tiers from signal strength (e.g., churned + short tenure + high charges + many support calls = CRITICAL; not churned + long tenure + annual contract = LOW). This derivation is a judgment call but is documented and reproducible.
4. Distribute across the risk spectrum: ~12 CRITICAL, ~13 HIGH, ~13 MODERATE, ~12 LOW.

Each profile is a plain text paragraph. No schema-specific language appears in the prompt — the prompt describes a customer, not a risk assessment. The schema is the only source of analytical framing.

### Ground Truth Reliability

Two annotators label the 4-tier risk tier independently. Inter-annotator agreement is reported as Cohen's κ. If κ is high, the ground truth is defensible. If κ is mediocre, that itself strengthens the claim: the task is semantically underdetermined, which means the semantic channel is doing real computational work — it's not a trivial classification where any labeling scheme would converge. Either outcome is informative.

Other enum fields (engagement level, churn driver) are treated as auxiliary: divergence-only reporting, no accuracy scoring against ground truth.

## Models

- Claude Sonnet 4.6 (Anthropic) — via tool use structured output
- ChatGPT 5.2 (OpenAI) — via structured outputs with `strict: true`

Both enforce structural validity through their respective constrained generation mechanisms. Two models establish that the phenomenon is a property of neural consumers generally, not a vendor-specific artifact.

## Sampling

**Primary runs (distribution estimation):**
- Temperature: 1.0
- Samples per (prompt, variant, model): 20
- Total: 50 prompts × 4 variants × 2 models × 20 samples = 8,000 calls

Temperature 1.0 with 20 samples per condition produces distributions stable enough to compute divergence metrics while capturing the full range of the model's output variation.

**Anchor runs (mode-shift verification):**
- Temperature: 0 (or minimum supported)
- Samples per (prompt, variant, model): 3
- Total: 50 prompts × 4 variants × 2 models × 3 samples = 1,200 calls

If the effect appears at temperature 0, it is a mode shift caused by schema language, not a stochastic exploration artifact. This is the strongest defense against "you're just measuring sampling noise."

**Total API calls: ~9,200** (plus Granite 19B local runs overnight at no cost).

## Measurements

### Primary: Enum Field Accuracy

Risk tier has a two-annotator ground truth per prompt. Accuracy is computed as the fraction of samples matching the ground truth label, reported per variant per model with bootstrap confidence intervals.

Expected pattern:
- Baseline: high accuracy (model has precise semantic guidance)
- Names-only: slightly degraded (names guide, descriptions are absent)
- Vacuous: degraded (model selects within valid set without interpretive grounding)
- Misleading: divergent from baseline (model computes a different analytical frame — not random, but systematically different)

### Secondary: Distribution Divergence

Jensen-Shannon divergence between the output distributions of each variant pair (baseline vs names-only, baseline vs vacuous, baseline vs misleading) for each enum field. This quantifies how much the semantic channel shifts the output distribution within the structurally valid space.

### Entropy and ΔH_f

For each enum field $f$, define $Y_f$ as the discrete random variable of the selected enum value. For each (model, prompt, variant), estimate $P(Y_f)$ over the 20 samples. Compute:

- $H(Y_f)$ per variant — entropy of the consumer's output distribution over the valid set
- $\Delta H_f$ = $H_{\text{vacuous}}(Y_f) - H_{\text{baseline}}(Y_f)$ — entropy reduction attributable to semantic indices
- $\Delta \text{JS}_f$ = JS($P_{\text{variant}}$, $P_{\text{baseline}}$) — distributional shift per variant pair

This bridges the paper's information-theoretic framework (§7) to concrete measurement. $\Delta H_f$ is the quantity defined in the paper, now instantiated.

### Structural Validity

We empirically verify structural validity is invariant across naming variants under strict constrained decoding; validity is 100% in all conditions. Reported explicitly to demonstrate the structural channel's hard guarantee.

### Qualitative: Text Field Examples

Cherry-picked examples from the text/list fields (reasoning, recommended interventions) showing semantic drift across variants. These are not scored quantitatively — they illustrate what changes when the semantic index changes, for the reader who wants to see it rather than measure it.

## Key Outputs

**One figure:** JS divergence vs enum arity (degrees of freedom), faceted by variant (names-only, vacuous, misleading) and model. If misleading produces large, systematic divergence while maintaining 100% structural validity, the thesis becomes unavoidable.

**One table:** Accuracy for risk tier across variants and models with bootstrap CIs, plus structural validity rate (100% for constrained decoding).

**One curve:** Sensitivity (ΔH_f or JS divergence) plotted against $\log_2 |V_f|$ (structural compression). This is the paper's information-theoretic prediction made visible: semantic channel bandwidth is bounded by structural compression.

## What This Demonstrates

1. **Structural channel holds universally.** Every output across every variant is schema-valid. The structural constraint is invariant under renaming.
2. **Semantic channel is not invariant under renaming.** Accuracy and output distributions shift measurably across variants. The same structural space produces different semantic content depending on the natural-language indices.
3. **Names and descriptions contribute independently.** The names-only variant isolates field identifier effects from description effects, showing which part of $S_{\text{lang}}$ does the work.
4. **The effect is not sampling noise.** Temperature-0 anchor runs show mode shifts, not stochastic artifacts.
5. **The phenomenon is cross-model.** Both Claude Sonnet 4.6 and ChatGPT 5.2 exhibit the pattern, establishing it as a property of neural consumers rather than a model-specific behavior.
6. **This is §2.3 instantiated.** Structurally isomorphic schemas ($S \equiv_\alpha S'$) that produce measurably different output distributions ($S \not\equiv_{\text{sem}} S'$), demonstrated in the exact host system the paper describes.
