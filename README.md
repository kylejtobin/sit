# Semantic Index Types

When a language model consumes a type schema, names stop being addresses and become instructions.

```python
# These two fields have the same type.
# They do not produce the same output.

churn_risk_tier: RiskTier   # → assesses voluntary customer departure risk
x7: RiskTier                # → picks an enum member
```

A **semantic index type** is a type declaration in which natural-language tokens — field names, descriptions, enum member names — function as computational indices that shape the output of a neural consumer. The name doesn't just label a slot. It tells the model what to compute.

## Why This Happens

Traditional types compile to machine code that erases names. The CPU has no use for `churn_risk_tier` — it sees a memory offset. Alpha equivalence holds because the compilation target doesn't read names.

Schema-driven types compile to token sequences consumed by a neural network that *reads* them. The compilation target changed. Names survive into the execution context, and the model interprets them as natural-language instructions. Alpha equivalence fails — not in the abstract, but because the new target architecture doesn't erase what the old one did.

## What the Paper Contributes

The [paper](semantic-index-types.md) formalizes this as a **two-channel constraint system**:

- The **structural channel** (type annotations, enum membership, validators, constrained decoding) bounds the space of valid outputs. It obeys alpha equivalence. It is mechanically enforced.
- The **semantic channel** (field names, descriptions, enum labels) guides the consumer's selection within that space. It violates alpha equivalence. It is neurally interpreted.

The interaction between the channels has **information-theoretic structure**: the semantic channel's effective capacity is bounded by the structural compression of the type. A `bool` gives 1 bit of semantic influence. A four-member enum gives 2 bits. A bare `str` gives unbounded capacity. This single quantity governs both the engineering design space and the security attack surface — the same variable controls how much a well-chosen name can help and how much an adversarial name can exploit.

The paper traces **converging empirical evidence** from three independently evolved research communities — schema-guided dialogue (SGD-X), text-to-SQL schema linking (Dr.Spider, BIRD), and code language model robustness (identifier obfuscation studies) — showing that each independently discovered what amounts to linguistic relativity for neural consumers: the vocabulary of the schema determines the output distribution of the model.

## Practical Consequences

- **Naming is programming.** Choosing `churn_risk_tier` over `attrition_risk_tier` is choosing between two analytical framings. It shapes what the model weighs, what thresholds it applies, what risk narrative it constructs.
- **Renaming is refactoring.** Empirical evidence shows renaming-class transformations produce 7–50% output degradation across benchmarks and model families. Renaming requires the same care as modifying function logic.
- **Descriptions are program text.** `Field(description=...)` content propagates into the schema the model reads. Removing a description changes the output distribution. It is part of the program.
- **Progressive hardening is the development methodology.** Start with semantic precision. Observe where the semantic channel fails. Promote those failures into structural guarantees — validators, tighter types, constrained decoding. Each step converts soft guidance into hard proof.
- **If names are instructions, they are also attack vectors.** Every field name and description is a point where the data/instruction boundary collapses — the same class of vulnerability that underlies SQL injection and XSS, instantiated at the schema level.

## Repository Contents

- [`semantic-index-types.md`](semantic-index-types.md) — the paper
- `sit/` — experiment code (in progress)

## License

Source code is MIT. Written content is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for details.
