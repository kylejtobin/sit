# Semantic Index Types

When a language model consumes a type schema, names stop being addresses and become instructions.

```python
# These two fields have the same type.
# They do not produce the same output.

churn_risk_tier: RiskTier   # → assesses voluntary customer departure risk
x7: RiskTier                # → picks an enum member
```

A **semantic index type** is a type declaration in which natural-language tokens — field names, descriptions, enum member names — function as computational indices that shape the output of a neural consumer. The name doesn't just label a slot. It tells the model what to compute.

## Five Principles

**1. Naming is programming.** Choosing `churn_risk_tier` over `attrition_risk_tier` is choosing between two analytical framings — voluntary departure versus passive loss. It shapes what the model weighs, what thresholds it applies, what risk narrative it constructs. The field name is an instruction. Changing the name changes the computation.

**2. Descriptions are program text.** `Field(description=...)` content propagates into the JSON Schema the model reads. A description that says "Projected total revenue across the full customer relationship, not historical sum" narrows the model's interpretation from a broad concept to a specific calculation. Removing the description changes the output distribution. It is part of the program.

**3. Renaming is refactoring.** In traditional programming, renaming a variable is a safe, mechanical operation — the behavior doesn't change. With language models consuming schemas, that invariant breaks. Empirical evidence shows renaming-class transformations produce 7–50% output degradation across benchmarks and model families. Renaming requires the same care as modifying function logic.

**4. Types constrain, names guide.** The type annotation bounds what the model can produce — a 4-member enum admits exactly 4 values. The field name and description guide which of those valid values the model selects. The tighter the type constraint, the less the name needs to do. A `bool` gives the name 1 bit of influence. A bare `str` gives it everything. This is why you want both: tight types for structural proof, precise names for semantic guidance.

**5. If names are instructions, they are also attack vectors.** Every field name and description is a point where the data/instruction boundary collapses — the same class of vulnerability that underlies SQL injection and XSS, instantiated at the schema level. An adversarial field description exploits the same channel a legitimate one uses for guidance.

## Why This Happens

Traditional types compile to machine code that erases names. The CPU has no use for `churn_risk_tier` — it sees a memory offset. Renaming is safe because the compilation target doesn't read names.

Schema-driven types compile to token sequences consumed by a neural network that *reads* them. The compilation target changed. Names survive into the execution context, and the model interprets them as natural-language instructions. Renaming breaks things — not because of a bug, but because the new target architecture treats names as computation.

## The Formal Framework

The [paper](semantic-index-types.md) makes this precise. It defines a **two-channel constraint system**:

- The **structural channel** (type annotations, enum membership, validators, constrained decoding) determines the *support* of the output distribution — the set of values the model can produce. Mechanically enforced. Invariant under renaming.
- The **semantic channel** (field names, descriptions, enum labels) determines the *conditional probabilities* within that support — which valid value the model selects. Neurally interpreted. Not invariant under renaming.

The interaction between the channels has **information-theoretic structure**. For a field $f$ with valid set $V_f$, the mutual information between the schema naming and the model's output is bounded:

$$I(N; Y_f) \leq H(Y_f) \leq \log_2 |V_f|$$

This single inequality governs both the engineering design space (how much a precise name helps) and the security attack surface (how much an adversarial name hurts). The paper defines a citable metric — **semantic index sensitivity** $d_f(S, S')$ — that directly measures the distributional shift when names change and structure doesn't.

**Progressive hardening** is the development methodology: start with semantic precision (good names, clear descriptions). Observe where the semantic channel fails. Promote those failures into structural guarantees — validators, tighter types, constrained decoding. Each step converts soft guidance into hard proof.

## The Experiment

We're running an experiment to measure this directly. Four structurally isomorphic Pydantic schemas — baseline (precise names), names-only (names without descriptions), vacuous (structural identifiers only), and misleading (coherent wrong-domain names) — applied to the same customer analysis task across multiple language models. The experiment measures $\Delta H_f$ (entropy reduction from semantic indices) and plots sensitivity against structural compression. Methodology is defined in [experiment.md](experiment.md); results are pending.

## Repository Contents

- [`semantic-index-types.md`](semantic-index-types.md) — the paper
- [`experiment.md`](experiment.md) — experiment design
- `sit/` — experiment code (in progress)
- [`.agents/scripts/building_block.py`](.agents/scripts/building_block.py) — recursive Pydantic type classifier (dev tool + TCA teaching example)

## License

Source code is MIT. Written content is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for details.
