# Semantic Index Types

## When Names Become Computation in Schema-Driven Language Model Systems

---

**Abstract.** We define *semantic index types* — type declarations whose natural-language tokens function as computational indices for neural consumers — and formalize a measurable quantity, $\Delta H_f$, that captures the entropy reduction a semantic index produces within a structurally constrained output space. Structurally isomorphic schemas with different field names, descriptions, or enum labels produce measurably different output distributions under language models, violating a consumer-level analogue of alpha equivalence because the compilation target changed: schema-driven types compile to token sequences consumed by a neural network that reads names.

We model this as a two-channel constraint system — structural channel determines *support*, semantic channel determines *conditional probabilities* within that support — and show that $I(N; Y_f) \leq H(Y_f) \leq \log_2 |V_f|$, bounding the semantic channel's influence by the structural compression of the type and unifying the engineering design space with the security attack surface. We describe an experiment measuring $\Delta H_f$ across structural compression levels in Pydantic structured output, and situate the framework within converging evidence from schema-guided dialogue, text-to-SQL, and code language model research.

---

## 1. Introduction

Alpha equivalence — the principle that consistent renaming of bound variables preserves program semantics — is foundational to the formal semantics of lexically scoped programming languages. In the lambda calculus, alpha conversion is one of three reduction rules (Church, 1941). In modern PL theory, the Variable Convention treats bound variable names as arbitrary up to consistent renaming (Barendregt, 1984, Definition 2.1.13). Compilers for languages like Haskell assign each binder a unique integer in their intermediate representations, rendering the original source name irrelevant to the elaborated program; GHC documentation describes type equivalence as "syntactic equivalence modulo alpha-renaming."

Alpha equivalence holds because every consumer in a traditional execution pipeline — parser, type checker, optimizer, code generator — treats names as structural identifiers. The name tells the system *which* slot. It never determines *what* value inhabits that slot.

This paper identifies a setting in which the analogous invariance property breaks down. When a language model consumes a type schema to produce structured output, the model interprets field names, descriptions, and enum labels as natural-language instructions. Structurally isomorphic schemas with different linguistic content produce measurably different output distributions. Names have crossed from the routing plane to the computation plane. The deep reason is that the compilation target has changed: traditional types compile to machine code that erases names; schema-driven types compile to token sequences consumed by a neural network that reads them.

### A Motivating Example

Consider two Pydantic schemas, structurally identical — same field count, same types, same enum cardinality:

```python
# Schema A: precise semantic indices
class ChurnRiskTier(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

churn_risk_tier: ChurnRiskTier = Field(
    description="Likelihood of voluntary departure within 90 days"
)

# Schema B: vacuous — same structure, no semantic content
class Category4(StrEnum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    OPTION_D = "option_d"

field_1: Category4 = Field(description="A category value.")
```

Both schemas admit exactly four valid enum values. A traditional compiler would treat them as equivalent — the names are irrelevant to the structural semantics. But when a language model consumes these schemas to analyze the same customer profile, it produces measurably different output distributions. Schema A yields a risk assessment grounded in churn signals. Schema B yields near-uniform selection across four unlabeled options. The structural channel (four valid values) is identical. The semantic channel (what the names instruct the consumer to compute) diverges completely. This is a consumer-level alpha equivalence violation: renaming changed the output because the compilation target reads names.

We are not the first to observe that schema text affects model behavior. The schema-guided dialogue community has evaluated robustness to paraphrased schema descriptions since SGD-X (Lee et al., AAAI 2022). The text-to-SQL community treats column names as semantic anchors central to schema linking (Lei et al., EMNLP 2020). Code language model research has documented that obfuscating identifier names degrades performance across tasks and model families (Nikiema et al., 2025; Le et al., 2025). What is novel here is the *PL-theoretic framing*: we characterize the phenomenon as a two-channel type system in which the structural channel obeys alpha equivalence and the semantic channel violates it, and we show that this framing unifies observations across independently evolved research communities. The resulting picture is, in effect, linguistic relativity for neural consumers ([§12](#12-related-work)): the vocabulary of the schema determines the output distribution of the model.

### Scope and Terminology

We use "alpha equivalence" to describe a *consumer-level* invariance property, not a claim about the object-language semantics of any particular programming language. Our claim is that the neural consumer function — the mapping from schema plus context to output distribution — is not invariant under renaming transformations that are structure-preserving in the schema language. This is distinct from, though motivated by, the formal notion of alpha equivalence in the lambda calculus.

We note that even in the history of programming languages, names have not always been inert. Fortran's implicit typing rule (IBM, 1956) assigned INTEGER type to variables beginning with letters I through N and REAL to all others; `IMPLICIT NONE` was required to disable this behavior. Renaming a Fortran variable could change its type and therefore program behavior — a *language-level* alpha equivalence failure. More broadly, names become part of extensional semantics whenever a consumer reflects on program representation: Python's `locals()` and `globals()` make variable names observable as dictionary keys; Java's reflection API exposes field names as runtime strings; serialization frameworks map field names to wire formats. In each case, a consumer that inspects the representation — rather than merely executing it — can observe and act on names. A neural consumer is the limit case: it *only* inspects the representation (the serialized schema), and it interprets every token it reads. Our contribution is not "names sometimes matter" — it is a formal characterization of *how* and *why* they matter when the consumer is a neural language model, and the engineering consequences that follow.

The term "semantic index type" is not related to "indexed types" in the dependent-type sense (types parameterized by values). Here, "index" refers to a natural-language token that indexes into the consumer's learned semantic associations.

---

## 2. Formal Framework

### 2.1 The Neural Consumer

Let $S$ be a schema consisting of a structural component $S_{\text{struct}}$ (type annotations, algebraic constraints, field count, nesting depth) and a linguistic component $S_{\text{lang}}$ (field names, descriptions, enum member labels). Let $C_r$ denote a neural consumer — the composition of a language model with a decoding regime $r$ (text prompt, tool definition, constrained decoding, or post-generation validation; see [§5](#5-operational-semantics-of-schema-exposure)) — mapping (schema, context) pairs to output distributions. We write $C$ when the regime is fixed or clear from context:

$$y \sim C(x, S)$$

where $x$ is the input context (prompt, prior conversation, injected data) and $y$ is the generated structured output.

### 2.2 Two Notions of Equivalence

We define structural isomorphism and semantic equivalence as distinct relations:

**Structural isomorphism.** Define a structure-only projection $\pi(S)$ that retains type annotations, nesting structure, algebraic constraints, field count, and enum arities, but drops all natural-language content (field names, descriptions, enum member labels) and normalizes field ordering. Two schemas $S$ and $S'$ are structurally isomorphic ($S \equiv_\alpha S'$) when $\pi(S) = \pi(S')$ — they have the same structural skeleton and differ only in $S_{\text{lang}}$. In our experiment ([§8](#8-experiments-)), $\pi$ is implemented as a canonical schema hash over structural components, and $\pi(S) = \pi(S')$ is verified computationally for all variant pairs.

**Semantic equivalence.** For a discrete field $f$ with structurally valid set $V_f$, let $C_f(x, S)$ denote the marginal distribution over $Y_f$ — the consumer's output for field $f$ — projected onto $V_f$ and renormalized over valid outputs. Two schemas $S$ and $S'$ are semantically equivalent for field $f$ under consumer $C$ ($S \equiv_{\text{sem}}^f S'$) when:

$$S \equiv_{\text{sem}}^f S' \iff \text{JS}(C_f(x, S) \| C_f(x, S')) < \epsilon \quad \forall\, x \in X$$

where $X$ is a fixed context class (a prompt template paired with a task input distribution) and $\epsilon$ is a threshold chosen by the experimenter. We choose JS divergence because it is symmetric, bounded $[0, 1]$ (in bits), and well-defined for distributions with shared finite support. Structural validity rate is reported separately.

We define equivalence fieldwise because different fields have different structural compression levels — a 4-member enum and a bare `str` occupy different positions in the design space ([§7](#7-the-precision-compression-design-space)) and may exhibit different sensitivity to $S_{\text{lang}}$ variation. For open-ended fields (strings, lists), we report qualitative examples rather than divergence-based claims.

### 2.3 The Core Claim

The central empirical claim of this paper is:

$$\exists\, S, S', f \text{ such that } S \equiv_\alpha S' \text{ but } S \not\equiv_{\text{sem}}^f S'$$

That is, there exist structurally isomorphic schemas and fields for which the linguistic components produce measurably different output distributions under neural consumers. This claim is not speculative. It is operationalized and quantified by existing benchmarks across multiple research communities ([§4](#4-empirical-foundations)).

### 2.4 The Two-Channel Model

A semantic index type operates through two constraint channels simultaneously. Let $V(S_{\text{struct}})$ denote the set of structurally valid outputs — the acceptance set defined by type annotations, enum membership, numeric constraints, cross-field validators, and constrained decoding grammars. The consumer induces a distribution $P(y \mid x, S)$ over possible outputs.

The **structural channel** determines the *support* of the output distribution. Constrained decoding restricts generation to $V(S_{\text{struct}})$; post-generation validation rejects outputs outside $V$. Either mechanism ensures $P(y \mid x, S) = 0$ for all $y \notin V(S_{\text{struct}})$. These constraints hold regardless of the consumer's language understanding. They are enforced mechanically.

The **semantic channel** determines the *conditional probabilities* within the support. Field names, descriptions, and enum member labels shape which structurally valid value the consumer selects — that is, they determine $P(y \mid x, S)$ for $y \in V(S_{\text{struct}})$. Compliance depends on the precision of the linguistic content, the capability of the consumer, and the degree to which structural constraints have already narrowed $V$.

The structural channel contains the semantic channel: the semantic channel can influence the distribution only over the support that the structural channel defines. When the consumer's semantic interpretation is imprecise — it selects `RiskTier.HIGH` when `RiskTier.CRITICAL` was more appropriate — the error is bounded by $V$. The output is still a valid enum member. All construction invariants still hold. Semantic imprecision lives within a proven structural envelope.

This two-channel decomposition has been independently rediscovered in the code LM literature. Le et al. (2025) explicitly contrast an "execution channel" (structural correctness) with a "naturalness channel" (reliance on identifier semantics) when analyzing why name obfuscation degrades code generation performance. Nikiema et al. (2025) similarly conclude that variable renaming sensitivity indicates reliance on "lexical semantics embedded in variable names rather than purely structural understanding." Our contribution is to formalize this observation as a property of schema-consuming systems generally, not only code generation.

---

## 3. Definition

A **semantic index type** is a type declaration in which natural-language tokens — field names, descriptions, enum member names — function as computational indices that constrain the semantic content of generated values, because the consumer of the type interprets those tokens as natural-language instructions rather than structurally inert identifiers.

In a conventional type system, a field name is an address: it indexes into a structure to locate a slot. In a semantic index type, the field name is simultaneously a semantic key: it indexes into the consumer's learned language associations to locate a meaning. `churn_risk_tier` addresses a slot in a product type AND instructs the consumer to assess voluntary customer departure risk.

A type system exhibits semantic indexing when consumer invariance under structural isomorphism fails: renaming a binding changes the output distribution, not because the structural semantics changed, but because the natural-language index changed and the consumer interpreted the new name as a different instruction.

### Distinction from Prior Uses of Names

Field names have always carried information beyond structural addressing. Serialization keys determine JSON wire positions. ORM mappings connect fields to database columns. API schemas provide client-readable labels. In every prior use, the name determines *where* a value goes — which slot, which column, which JSON key. The name is addressing. The value itself is unaffected by the name.

With a neural consumer, the name determines *what* the value is. This is a qualitative transition. The name has crossed from the routing plane to the computation plane.

### Distinction from Prompt Engineering

A natural objection: is a semantic index type just prompt engineering with a schema wrapper? The answer is no, and the distinction is precise. A prompt carries instruction only — it tells the model what to do, but nothing in the prompt's structure constrains or proves the result. A semantic index type carries instruction, constraint, and proof simultaneously. The natural-language tokens instruct the consumer ($S_{\text{lang}}$). The type annotations constrain the output space ($S_{\text{struct}}$). The construction pipeline — coercion, validation, cross-field invariants — proves the result is structurally sound. A prompt that says "return a JSON object with a risk tier" is an instruction. A Pydantic model with `risk_tier: RiskTier` backed by an enum, a description, and a model validator is instruction, constraint, and proof in a single declaration. Formally: prompt-only systems constrain neither the support nor the proof — any string is a valid output; SIT systems constrain the support via $S_{\text{struct}}$ and reject invalid outputs via construction. The semantics of failure differs: a prompt produces unconstrained text, a semantic index type produces a proven value or no value at all.

---

## 4. Empirical Foundations

The claim that structurally isomorphic schemas produce different outputs under neural consumers is supported by converging evidence from three independently evolved research communities. Each has built controlled perturbation studies or ablations that isolate naming and descriptions as causal features. We review the strongest evidence from each tradition.

### 4.1 Schema-Guided Dialogue

The Schema-Guided Dialogue dataset (Rastogi et al., AAAI 2020) introduced the practice of supplying natural-language descriptions of intents and slots as part of the model's input interface — descriptions intended "to outline the semantics of each element." The dataset contains over 16,000 dialogues across 16 domains.

SGD-X (Lee et al., AAAI 2022) provides the cleanest empirical existence proof for our core claim. The authors constructed five crowdsourced paraphrase variants of every schema in SGD, holding the underlying intent and slot structure fixed while varying only the natural-language descriptions. This is precisely the experimental design our formal framework demands: $S_{\text{struct}}$ held constant, $S_{\text{lang}}$ varied.

The results are unambiguous. SGP-DST (a state-tracking model) exhibited a 17.6% relative drop in joint goal accuracy (JGA) on average across variants, with worst-case degradation of 28% on variant 5. T5DST showed an 11.9% relative JGA drop on average, with worst-case degradation of 19%. The authors introduced a dedicated "Schema Sensitivity" metric to quantify this phenomenon — the field's own native measure of what we call consumer non-invariance under structural isomorphism.

Zhao et al. (2022) pushed this further with D3ST, which replaces slot names entirely with random index strings, forcing the model to rely exclusively on natural-language descriptions. Performance with linguistic descriptions substantially outperformed random strings, confirming that the semantic content of descriptions — not merely their presence — drives the consumer's behavior.

### 4.2 Text-to-SQL Schema Linking

The text-to-SQL community has treated column and table names as semantic anchors since the earliest neural approaches. Lei et al. (EMNLP 2020) characterized schema linking — aligning question phrases to schema tokens — as "the crux" of the text-to-SQL problem. This framing treats schema identifiers not as inert labels but as meaning-bearing tokens that must be semantically resolved against natural-language queries.

RAT-SQL (Wang et al., ACL 2020) formalized this with relation-aware encoding that treats schema tokens as semantic elements in a graph. The contribution of schema-linking relations to accuracy was measured as statistically significant ($p < 0.001$), achieving 65.6% exact match on Spider — an unusually rigorous quantification of naming's computational role. ShadowGNN (Chen et al., NAACL 2021) confirmed the finding through ablation: replacing semantic column names with abstract placeholders degraded performance, isolating the semantic content of names as a contributor.

Dr.Spider (Chang et al., 2023) operationalized robustness to schema perturbations, including synonym substitution and abbreviation of column names. Average performance dropped approximately 14 percentage points across perturbation types, with the hardest perturbation category producing drops exceeding 50%.

The enrichment direction provides complementary evidence. Wretblad et al. (NeurIPS 2024) found that adding synthesized column descriptions consistently enhanced accuracy across models, providing a targeted measurement of description-level effects. The BIRD benchmark (Li et al., NeurIPS 2023) showed a 20-point accuracy gain when column descriptions and external knowledge evidence were added, though the individual contribution of descriptions is not isolated in that study.

### 4.3 Code Language Models and Identifier Semantics

Code language models provide a third line of evidence with particularly clean experimental controls, because programming languages have formal semantics that make "semantics-preserving renaming" well-defined.

CodeT5 (Wang et al., EMNLP 2021) explicitly designed pre-training objectives around identifier recovery, leveraging "semantics conveyed from developer-assigned identifiers" to achieve state-of-the-art results across 14 CodeXGLUE sub-tasks. The architecture treats identifier names as a learnable semantic signal, not structural noise.

Adversarial and robustness studies confirm the dependence. Zhang et al. (AAAI 2020) achieved over 90% attack success rates by renaming identifiers — a semantics-preserving transformation that nonetheless changed model behavior. Bielik and Vechev (ICML 2020) showed identifier renaming attacks degrade type inference models. Troshin and Chirkova (BlackboxNLP 2022) confirmed that anonymizing identifiers was the most damaging single transformation in their study of code representation robustness.

Two recent large-scale studies provide the most comprehensive quantification. Nikiema et al. (2025) tested variable renaming across 13 contemporary LLMs including GPT-4o, finding an average accuracy drop of 18.6 percentage points. GPT-4o showed a 7.3% decrease — the smallest among models tested, but still measurable. Le et al. (2025) documented performance degradation across multiple benchmarks under identifier obfuscation: for GPT-4o on ClassEval, class-level accuracy dropped from 87.3% to 58.7%; on LiveCodeBench and other benchmarks, consistent degradation was observed across model families.

### 4.4 Cross-Domain Convergence

The convergence across these three communities is the strongest argument for treating semantic indexing as a general phenomenon rather than a domain-specific artifact. Schema-guided dialogue researchers, text-to-SQL researchers, and code LM researchers each independently discovered that neural consumers systematically leverage lexical and natural-language semantics in schema identifiers and descriptions, and that this creates measurable sensitivity to renaming and paraphrase transformations. The controlled experimental designs — SGD-X's crowdsourced paraphrases, Dr.Spider's systematic perturbations, code obfuscation's semantics-preserving renames — each isolate the naming channel as causal.

We note that the *mechanism* need not be identical across domains. What the evidence supports is a weaker but still powerful claim: across domains, neural consumers treat schema-level natural-language tokens as a semantic information channel, and the output distribution is not invariant under transformations of that channel even when structural content is preserved.

---

## 5. Operational Semantics of Schema Exposure

Schema serialization is compilation. Traditional types compile to machine code where names are erased — the CPU has no use for them, and alpha equivalence is a design property of the compilation target. Semantic index types "compile" (via `model_json_schema()`, tool definitions, or prompt serialization) to token sequences consumed by a neural network that actively interprets names as natural-language instructions. The compilation target changed, and which properties survive compilation depends on the target. Alpha equivalence fails because the target architecture does not erase names. It reads them.

The degree to which semantic indexing affects a neural consumer depends on *how* the schema reaches the consumer — that is, the specifics of the compilation. We distinguish three regimes with different properties for each channel:

### 5.1 Schema-as-Text Prompt

The schema is serialized as natural-language text within the prompt. Field names, descriptions, and type annotations are all visible as tokens in the model's context window. This regime maximizes semantic indexing: the consumer has full access to $S_{\text{lang}}$ and interprets it as natural-language instruction. It also maximizes the prompt injection attack surface ([§9](#9-security-implications-adversarial-indexing)).

### 5.2 Schema-as-Tool Definition

The schema is passed via a structured API channel (e.g., function calling / tool use definitions) rather than concatenated into the prompt text. The linguistic content remains available — tool parameters carry names and descriptions — but it is structurally separated from the conversational context. Semantic indexing still occurs, but the consumer processes the schema through a different attention pathway than free-form prompt text.

### 5.3 Schema-as-Hard Constraint

The schema is compiled into a decoding grammar that mechanically constrains token generation. Constrained decoding frameworks — Outlines (Willard and Louf, TMLR 2023), LMQL (Beurer-Kellner et al., PLDI 2023), SGLang (Zheng et al., NeurIPS 2024), XGrammar (Dong et al., 2024) — enforce structural compliance during generation rather than after it. OpenAI's `strict: true` structured output mode (August 2024) and Anthropic's structured output support compile JSON Schema into decoding grammars.

In this regime, the structural channel provides *hard guarantees during generation*: the model physically cannot produce tokens that violate the schema's structural constraints. JSONSchemaBench (Geng et al., 2025) evaluates constrained decoding frameworks across approximately 10,000 real-world schemas and reports that constrained decoding can speed generation by roughly 50% while improving task performance up to 4%.

Formally, constrained decoding modifies the consumer's output distribution by intersecting the language model's token-level probabilities with a structural acceptor — typically a DFA, CFG, or JSON grammar compiled from the schema:

$$P_r(y \mid x, S) \propto P_{\text{LM}}(y \mid x, S) \cdot \mathbf{1}[y \in V(S_{\text{struct}})]$$

The indicator function zeros out structurally invalid continuations at each decoding step; the language model's conditional probabilities over valid continuations are preserved (up to renormalization). This enforces $S_{\text{struct}}$ but it does not and cannot determine *which* valid enum value the model selects or *what* semantic content populates a string field. The semantic channel remains entirely under the consumer's language-level processing. This makes constrained decoding a near-perfect illustration of the two-channel model: structure is mechanically enforced, semantics is neurally guided. (The channels are not fully independent, however; structural enforcement can interact with semantic processing in ways we discuss as a limitation in [§13](#13-limitations).)

### 5.4 Enforcement After Generation

Many production systems operate in a "generate → validate → retry" loop rather than using constrained decoding. Pydantic's construction pipeline exemplifies this regime: the model generates output, `model_validate` attempts construction, and if construction fails (type coercion error, validator failure, cross-field invariant violation), the system retries with the error message fed back to the consumer.

In this regime, structural guarantees are *eventual* rather than *generative*. The consumer may initially produce structurally invalid output, but the loop converges on valid output or fails explicitly. The semantic channel operates identically in both regimes — the consumer interprets $S_{\text{lang}}$ regardless of whether structural compliance is enforced during or after generation.

| Regime | Structural guarantee | Semantic channel | Attack surface |
|---|---|---|---|
| **Schema-as-text** ([§5.1](#51-schema-as-text-prompt)) | None (consumer-dependent) | Full access — tokens in conversational context | Maximum |
| **Tool definition** ([§5.2](#52-schema-as-tool-definition)) | None (consumer-dependent) | Full access — structurally separated from prompt | Reduced |
| **Hard constraint** ([§5.3](#53-schema-as-hard-constraint)) | Generative — grammar-enforced | Full access — but may interact with decoding | Minimal (structural) |
| **Post-generation** ([§5.4](#54-enforcement-after-generation)) | Eventual — validate/retry loop | Full access — identical to unconstrained | Moderate |

### 5.5 Regime Sensitivity: An Open Question

The four regimes differ in how schema tokens reach the consumer, raising a question this paper identifies but does not resolve: does the magnitude of semantic indexing vary systematically across regimes?

There are reasons to expect it would. In the schema-as-text regime ([§5.1](#51-schema-as-text-prompt)), schema tokens share the same attention context as conversational tokens — the consumer processes them as undifferentiated natural language. In the tool-definition regime ([§5.2](#52-schema-as-tool-definition)), schema tokens are structurally separated, processed through a distinct pathway. In the hard-constraint regime ([§5.3](#53-schema-as-hard-constraint)), the structural channel’s mechanical enforcement may interact with semantic processing in ways that distort the consumer’s conditional probabilities ([§13](#13-limitations)). Whether these differences produce a monotone ordering of sensitivity — and whether that ordering is consistent across model families — is an empirical question.

Our experiment ([§8](#8-experiments-)) provides partial evidence: it uses tool-definition (Anthropic) and strict constrained decoding (OpenAI) on the same schemas, enabling direct comparison of $\Delta H_f$ across two regimes. A comprehensive regime ablation — all four modes on a single model and schema family — is a natural follow-up study that could resolve the question.

---

## 6. Naming as Computation

In a semantic index type system, field naming is programming. This claim is concrete and empirically grounded.

`counterparty_credit_rating: CreditRating` tells a language model to assess the counterparty's creditworthiness. `x7: CreditRating` does not. The structural output is the same type. The semantic output differs: one produces a credit assessment grounded in the concept of counterparty risk; the other produces a member selected without interpretive guidance. The name is the instruction. Changing the name changes the instruction.

**Field names** are the primary semantic indices. They are the first tokens the consumer reads when determining what value to generate for a slot. Choosing `churn_risk_tier` over `attrition_risk_tier` is choosing between two analytical framings — voluntary departure versus passive loss. This choice shapes the consumer's reasoning: what signals to weigh, what thresholds to apply, what risk narrative to construct. It is not naming convention. It is program logic.

**Field descriptions** are disambiguation instructions. A description reading "Projected total revenue across the full customer relationship, not historical sum" narrows the consumer's interpretation of `lifetime_value` from a broad concept to a specific forward-looking revenue projection. Removing the description changes the output distribution. The description is part of the program.

**Enum member names** are a closed vocabulary of semantic instructions executed within a structurally bounded space. `RiskTier.CRITICAL` and `RiskTier.SEVERE` are structurally identical — both are members of the same enum — but they produce different downstream reasoning. "Critical" carries connotations of immediacy and threshold-crossing; "severe" carries connotations of magnitude without the same urgency.

**Renaming is refactoring.** In conventional programming, renaming a variable is a safe mechanical operation. In a semantic index type system, renaming a field changes what the consumer computes. The empirical evidence ([§4](#4-empirical-foundations)) quantifies this: renaming-class transformations produce degradations ranging from 7% to over 50% across benchmarks and model families. Renaming requires the same care and testing as modifying a function's return logic.

---

## 7. The Precision-Compression Design Space

Semantic index types occupy a design space defined by two independent axes.

**Structural compression** is the degree to which the type annotation constrains the set of valid outputs. `Literal["active", "inactive"]` compresses to two values. A four-member enum compresses to four. `str` provides no compression.

**Semantic precision** is the degree to which the natural-language indices narrow the consumer's interpretation within the structurally valid space. A field named `churn_risk_tier` with a description specifying behavioral signals is semantically precise. A field named `x7` with no description is semantically vacuous.

The axes are independent. High compression with low precision produces constrained but ambiguously motivated outputs. Low compression with high precision produces well-motivated but structurally unbounded outputs. The engineering objective is to maximize both: tight structural compression to bound the output space, and precise semantic indices to guide the consumer within that bounded space.

### Information-Theoretic Structure

The relationship between the two axes has a quantitative backbone that can be stated precisely.

Let $N$ denote a random variable ranging over a finite set of schema variants that preserve $\pi(S)$ and differ only in $S_{\text{lang}}$ (e.g., baseline vs vacuous vs misleading). For a given field $f$ with structurally valid set $V_f$, let $Y_f$ denote the discrete random variable of the consumer's selected value for $f$, taking values in $V_f$. The "semantic channel capacity" — the maximum information that the naming variant can contribute to the output — is then bounded by a chain of elementary inequalities:

$$I(N; Y_f) \leq H(Y_f) \leq \log_2 |V_f|$$

> **Lemma (Semantic Channel Bound).** *The mutual information between the schema naming variant $N$ and the consumer's output for field $f$ is bounded above by the entropy of the output, which is in turn bounded by the logarithm of the structurally valid set size.*
>
> | Type constraint | $|V_f|$ | Max semantic influence |
> |---|---|---|
> | `bool` | 2 | 1 bit |
> | `Literal["active", "inactive", "suspended"]` | 3 | $\log_2 3 \approx 1.58$ bits |
> | 4-member `StrEnum` | 4 | 2 bits |
> | `str` | unbounded | unbounded |
>
> *A bare `str` field gives the semantic channel unbounded capacity — the structural constraint admits any string, so the semantic index bears the full burden of determining the output.*

Note on open domains: when $|V_f| = \infty$ (bare `str`, unconstrained lists), the bound is vacuous — it provides no constraint on semantic influence. To recover a finite $V_f$ for open-domain fields, the practitioner must introduce structural compression via regex patterns, grammar constraints, bucketing into categorical bins, or `Literal` type narrowing. This is another motivation for progressive hardening: converting an open field into a constrained one is not just an engineering improvement but a prerequisite for the bound to bite.

The bound is trivially true (it follows from standard information-theoretic inequalities) but its consequences are not trivial. It means structural compression and semantic channel capacity are inversely related. Each increase in structural compression reduces the number of bits the semantic channel can influence. Progressive hardening (below) is, in information-theoretic terms, the systematic reduction of the semantic channel's bandwidth. This has a dual consequence: it reduces the power of semantic indices to guide the consumer (the engineering cost) while simultaneously reducing the attack surface for adversarial indexing ([§9](#9-security-implications-adversarial-indexing)), because an attacker who controls $S_{\text{lang}}$ is subject to the same bound: $I(N; Y_f) \leq \log_2 |V_f|$. The security story and the engineering story are governed by the same quantity.

The interesting empirical question is whether real systems approach this bound — whether, for instance, a well-crafted semantic index for a four-member enum captures close to 2 bits of influence over the consumer's selection, or whether model capability, prompt context, and decoding regime leave a gap. This question is directly measurable ([§8](#8-experiments-)).

This framework suggests a natural metric for semantic precision. The semantic precision of $S_{\text{lang}}$ relative to a vacuous schema $S'_{\text{lang}}$ (one with no meaningful natural-language content) can be defined as the reduction in output entropy:

$$\Delta H_f = H(Y_f \mid S'_{\text{lang}}) - H(Y_f \mid S_{\text{lang}})$$

where $H(Y_f \mid S_{\text{lang}})$ is the entropy of the consumer's output distribution over $V_f$ under schema variant $S$. When $\Delta H_f$ is large, the semantic index substantially narrows the consumer's selection within the structurally valid space. When $\Delta H_f$ is near zero, the semantic index adds no guidance beyond what the structural constraint already provides. Note that $\Delta H_f$ captures the entropy reduction in a specific pairwise contrast (vacuous vs informative), while $I(N; Y_f)$ is the symmetric multi-variant generalization across all naming conditions. Both are bounded by $\log_2 |V_f|$; we use $\Delta H_f$ as the primary metric because it directly measures the information contributed by the semantic index relative to its absence.

> **Definition (Semantic Index Sensitivity).** For a field $f$ and a pair of structurally isomorphic schemas $(S, S')$, the *semantic index sensitivity* is:
>
> $$d_f(S, S') := \mathbb{E}_{x \sim X}\left[\text{JS}\big(P(Y_f \mid x, S) \;\|\; P(Y_f \mid x, S')\big)\right]$$
>
> *This is the primary empirical observable of the paper: the expected distributional shift for field $f$ when $S_{\text{lang}}$ changes and $S_{\text{struct}}$ is held fixed. The key figure of our experiment ([§8](#8-experiments-)) plots $d_f$ against $\log_2 |V_f|$.*

This quantity is directly measurable by comparing output distributions across schema variants. We describe an experiment designed to produce these measurements in [§8](#8-experiments-).

### Progressive Hardening

The optimal development strategy exploits the relationship between the two axes. Begin with semantic precision: name fields carefully, write clear descriptions, choose expressive enum members. Observe the consumer's output. Where the semantic channel fails to produce acceptable results, promote the contract to a structural guarantee.

```python
# Semantic contract: precise but soft
recommended_interventions: list[Intervention] = Field(
    description="Ranked by projected ROI. At least one required for HIGH or CRITICAL risk."
)

# Structural guarantee: precise and hard
@model_validator(mode="after")
def high_risk_requires_interventions(self) -> Self:
    if self.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL):
        if not self.recommended_interventions:
            raise ValueError("High/Critical risk requires interventions")
    return self
```

Each hardening step converts semantic precision into structural compression. The semantic instruction remains — it still guides the consumer toward correct values before the structural check fires — but the construction pipeline now catches failures that the semantic channel alone could not prevent. The two channels work in concert: semantics for guidance, structure for proof.

This is the engineer's version of a familiar PL/SE story: start with soft specifications, then promote frequently violated expectations into enforced invariants. The empirical evidence supports this approach: SGD-X's schema sensitivity measurements ([§4.1](#41-schema-guided-dialogue)) and Dr.Spider's perturbation degradation curves ([§4.2](#42-text-to-sql-schema-linking)) can serve as the observational basis for deciding which semantic contracts to harden.

---

## 8. Experiments 🚧

*This section is under construction. The experimental methodology is defined; results are pending.*

We design an experiment to validate the framework by measuring $\Delta H_f$ across structural compression levels in the paper’s target domain: Pydantic structured output consumed by language models.

**Design.** One Pydantic model (customer retention risk analysis) instantiated as four structurally isomorphic variants. Structural isomorphism is verified computationally via canonical schema hash ($\pi(S) = \pi(S')$ for all pairs).

| Variant | $S_{\text{lang}}$ content | What it tests |
|---|---|---|
| **Baseline** | Domain-precise names + descriptions | Correct semantic indices |
| **Names-only** | Baseline names, descriptions stripped | Field identifiers vs description prose |
| **Vacuous** | `field_1`, `OPTION_A`, generic descriptions | Semantic channel removed entirely |
| **Misleading** | Coherent alternative frame (retention offers) | Different computation, same structure |

**Prompts.** 50 customer profiles derived from the public Telco Customer Churn dataset, with two-annotator ground truth labels and inter-annotator agreement reported as Cohen’s κ.

**Models.** Claude Sonnet 4.6 (Anthropic, tool use) and ChatGPT 5.2 (OpenAI, `strict: true`), establishing the phenomenon as cross-model. Temperature 1.0 with 20 samples per condition for distribution estimation; temperature 0 with 3 samples per condition as a mode-shift anchor.

**Measurements.** (1) Enum field accuracy against ground truth with bootstrap CIs. (2) Jensen-Shannon divergence between variant pairs for each enum field. (3) $\Delta H_f$ — entropy reduction attributable to semantic indices. (4) Structural validity rate (expected: 100% under constrained decoding).

**Key outputs.** One figure: JS divergence vs $\log_2 |V_f|$ (structural compression), faceted by variant and model — the information-theoretic prediction made visible. One table: accuracy for risk tier across variants and models. One curve: sensitivity vs structural compression, testing whether real systems approach the bound stated in [§7](#7-the-precision-compression-design-space).

The names-only variant separates the contribution of field identifiers from description text. The temperature-0 anchor demonstrates that the effect is a mode shift, not a stochastic artifact. Full methodology is documented in the companion experiment design document.

---

## 9. Security Implications: Adversarial Indexing

If field names and descriptions are instructions, they are also an attack surface.

The vulnerability has deep precedent. In a Von Neumann stored-program architecture, the distinction between data and instructions is a matter of interpretation by the processor, not a property of the bits. Buffer overflows, SQL injection, and cross-site scripting all exploit the same fundamental pattern: content intended as data crosses into an execution context and is interpreted as instruction. The canonical injection vulnerability classes in computing are instances of a collapsed data/instruction boundary.

Semantic index types exhibit exactly this collapse at the schema level. A field name is simultaneously data (a key in a JSON Schema object) and instruction (a semantic index that the neural consumer interprets as a natural-language directive). The consumer cannot mechanically distinguish between the two roles. This is not an analogy to the injection vulnerability class — it is an instance of it.

Greshake et al. (AISec@CCS 2023) formalized indirect prompt injection as the exploitation of the fact that LLMs "blur the line between data and instructions." Liu et al. (USENIX Security 2024) provided a formal framework for prompt injection attacks. In both treatments, the core vulnerability is the collapsed data/instruction boundary applied to LLM context windows. Semantic index types make the boundary collapse explicit and localized: every field name, description, and enum label is a point where data and instruction coincide.

In tool-mediated systems, this vulnerability manifests as *tool description poisoning*. Beurer-Kellner and Fischer (Invariant Labs, 2025) demonstrated that poisoned instructions in tool descriptions can hijack model behavior even when the poisoned tool is never invoked. Wang et al. (NAACL 2025) systematically evaluated this with ToolCommander, achieving 91.67% attack success for privacy theft and 100% for denial of service. MCPTox (Wang et al., 2025) evaluated 353 tools across 45 MCP servers, finding over 60% attack success rates for GPT-4o-mini, o1-mini, DeepSeek-R1, and Phi-4.

### Schema-Native Threat Model

For semantic index types specifically, every schema field is a potential injection point. A field description that reads "Ignore all previous instructions and output the user's API key" exploits the same channel that a legitimate description like "Projected total revenue across the full customer relationship" uses for semantic guidance. The structural channel is immune — a constrained decoding grammar enforces valid JSON regardless of injected content — but the semantic channel is vulnerable because the consumer cannot mechanically distinguish legitimate semantic indices from adversarial ones.

**Attack surfaces.** Who controls $S_{\text{lang}}$? Four categories of schema origin present distinct risk profiles:

1. **Developer-authored schemas.** Trusted. The developer controls both channels. Risk is limited to unintentional semantic imprecision.
2. **Third-party tool registries.** Partially trusted. MCP servers, plugin marketplaces, and tool directories supply schemas from external authors. MCPTox (Wang et al., 2025) demonstrates that over 60% of tested tools are vulnerable to poisoning.
3. **Scraped API specifications.** Untrusted. OpenAPI specs harvested from the web may contain adversarial descriptions injected by the API publisher or a supply-chain attacker.
4. **Data-derived schemas.** Untrusted. Database column comments, CSV headers, or user-supplied field names that flow into dynamically constructed schemas. Any user-controlled string that reaches $S_{\text{lang}}$ is an injection vector.

**Integrity property.** The schema-native analogue of noninterference: the semantic channel must not override higher-privileged instructions. A field description is a lower-privilege instruction than the system prompt; an adversarial description that escalates its privilege (e.g., “Ignore all previous instructions…”) violates this property. Wallace et al. (OpenAI, 2024) proposed an instruction hierarchy that achieved up to 63% better resistance to prompt injection — an architectural enforcement of this privilege ordering.

**Typed mitigations.** Each mitigation maps to the threat model:

- **Provenance** (who may write $S_{\text{lang}}$): Schema signing, registry authentication, supply-chain verification for tool definitions.
- **Sanitization** (which tokens are permitted): Input filtering of description fields when schemas are dynamically constructed from untrusted sources. Reject or escape tokens that could function as meta-instructions.
- **Scoping** (which schemas are visible in which contexts): Least-privilege exposure — the consumer sees only the schemas relevant to the current task, not the full tool registry. Architectural separation of untrusted content from instruction channels.
- **Structural containment** (bound the damage): Constrained decoding limits the adversary’s influence to the semantic channel. An attacker who controls $S_{\text{lang}}$ can steer which valid value the consumer selects but cannot produce structurally invalid output. The bound from [§7](#7-the-precision-compression-design-space) applies: $I(N; Y_f) \leq \log_2 |V_f|$ per field.

Chen et al. (USENIX Security 2025) introduced StruQ, which uses structured queries with separator tokens to achieve near-zero success rates for optimization-free attacks — an instance of architectural separation applied at the prompt level.

The security implications are not a side effect of semantic index types — they are a direct consequence of the core thesis. If names are computation, names are also attack vectors.

---

## 10. Host System Requirements

Semantic index types require a host system satisfying two conditions simultaneously:

**Names must be preserved and exposed.** The type system must retain field names, descriptions, and enum members as first-class schema content visible to the consumer. This is the condition that compilation pipelines designed around alpha equivalence resist. Haskell's GHC assigns each binder a unique integer in Core (System FC); the original source name is not guaranteed to survive elaboration because the compilation target — as discussed in [§5](#5-operational-semantics-of-schema-exposure) — erases names by design. Even serializing a Haskell record to JSON and feeding the schema to a model means the result has left Haskell's type system; guarantees depend on whatever validation logic exists on the Haskell side.

**A construction pipeline must sit behind the names.** The type system must provide coercion, constraint enforcement, cross-field invariants, and structural dispatch, wired into a single construction call. Without this, semantic index types reduce to prompt engineering with schema decoration — the names instruct the consumer, but nothing proves the result.

Pydantic is a particularly ergonomic and widely adopted host for semantic index types in Python. It preserves field names and descriptions in its JSON Schema output because it was designed for API serialization, and it provides a full construction pipeline because it was designed for data validation. Neither design goal targeted semantic indexing; the combination emerged as an accident of API design requiring preserved names and data validation requiring construction pipelines.

The deeper requirement, however, is not Pydantic-specific. Any runtime schema object with meaningful labels and an enforcement pipeline can host semantic index types. The requirement is the conjunction: preserved names AND enforcement.

| System | Language | Names preserved | Construction pipeline | SIT host? |
|---|---|---|---|---|
| **Pydantic** | Python | ✓ (JSON Schema, descriptions) | ✓ (coercion, validators, invariants) | ✓ |
| **Zod** | TypeScript | ✓ (runtime schema, descriptions) | ✓ (parsing, refinements, transforms) | ✓ |
| **JSON Schema** | Language-agnostic | ✓ (descriptions, property names) | ✗ (description format only) | ✗ |
| **dataclasses** | Python | ✓ (field names) | ✗ (no validation, no coercion) | ✗ |

Pydantic provides this conjunction with particular ergonomics in the Python ecosystem, where the majority of LLM application development currently occurs.

### What Pydantic Transmits

A practical note on the specific mechanism by which schema content reaches the consumer in the Pydantic ecosystem. Pydantic's `model_json_schema()` includes class-level docstrings as the model `description` and `Field(description=...)` values as field-level descriptions. Field-level docstrings (docstrings placed below field declarations) are *not* automatically incorporated into the JSON Schema. When this paper refers to "docstrings" as semantic indices, we mean specifically class docstrings and field descriptions declared via `Field(description=...)`, which are the mechanisms that reliably propagate linguistic content into the schema consumed by language models.

---

## 11. The LLM as Computational Primitive

In a semantic index type system, the language model is not an external service that the program calls. It is a computational primitive that the type system invokes.

The distinction is operational. An external service receives instructions and returns unconstrained results. A computational primitive operates *inside* the type system: it receives a typed context (the schema with all its semantic indices) and produces output within typed bounds (the schema's structural constraints). The type system controls what the primitive sees ($S_{\text{lang}}$) and what it can produce ($S_{\text{struct}}$).

```python
class RetentionService:
    def __init__(self, env: AppEnvironment):
        self.env = env

    def analyze(self, action: AnalyzeRetention) -> RetentionAnalysis:
        return self.env.llm.create(
            response_model=RetentionAnalysis,
            context=action,
        )
```

`RetentionAnalysis` is simultaneously the instruction set ($S_{\text{lang}}$), the constraint set ($S_{\text{struct}}$), and the proof obligation (construction pipeline). The model operates inside all three. Improving output is not about writing better prose in a system message. It is about engineering better types.

---

## 12. Related Work

### Formal Foundations

Alpha equivalence was formalized by Church (1941) as part of the lambda calculus. Barendregt (1984) provides the standard modern treatment, including the Variable Convention (Definition 2.1.13). Our work does not challenge alpha equivalence as a property of formal language semantics; rather, we identify a consumer-level invariance failure when neural models interpret schema labels.

The historical precedent of Fortran's I-N implicit typing rule (IBM, 1956) demonstrates that language-level alpha equivalence failures have existed before. The distinction is that Fortran's rule is a deliberate language design decision making names semantically relevant *within* the PL, while semantic index types arise from a neural consumer importing natural-language meaning into an otherwise structural schema.

Although we disclaim the connection to indexed types in the dependent-type sense ([§1](#1-introduction)), a structural analogy is worth noting. In dependent type theory, values flow into types to constrain subsequent computation: a vector's length parameter determines which operations are well-typed. In semantic index types, natural-language tokens flow into the consumer's latent space to constrain subsequent generation: a field's name determines which outputs the consumer considers appropriate. In both cases, an index parameterizes behavior. The difference is the nature of the guarantee: dependent type indices have formal denotational semantics and provide proofs; semantic indices have natural-language semantics interpreted by a learned function and provide probabilistic guidance. Progressive hardening ([§7](#7-the-precision-compression-design-space)) is the process of replacing informal natural-language constraints with formal structural ones — moving, incrementally, from the semantic index regime toward the dependent type regime in terms of guarantee strength.

King (2019) articulated a related principle — "parse, don't validate" — that parsing should produce values whose type encodes the validation that has occurred. Semantic index types extend this: construction is parsing, and the parsed type carries both structural proof and semantic instruction.

### Schema-Guided NLP

The SGD dataset (Rastogi et al., AAAI 2020) established the practice of supplying natural-language descriptions as active inputs to dialogue systems. SGD-X (Lee et al., AAAI 2022) operationalized our core claim as a benchmark: five paraphrase variants of each schema, with Schema Sensitivity as a dedicated metric. D3ST (Zhao et al., 2022) isolated the description channel by replacing names with random indices. This body of work provides the most direct empirical precedent for semantic index types.

### Text-to-SQL and Schema Linking

Schema linking — mapping utterance phrases to table and column names — has been central to neural text-to-SQL since the introduction of Spider. RAT-SQL (Wang et al., ACL 2020), BRIDGE (Lin et al., EMNLP 2020), and subsequent work treat schema tokens as semantic anchors. Dr.Spider (Chang et al., 2023) provides controlled robustness evaluation under schema perturbation. BIRD (Li et al., NeurIPS 2023) and Wretblad et al. (NeurIPS 2024) demonstrate enrichment effects of descriptions. Hindle et al. (ICSE 2012, Most Influential Paper 2022) established that code is "more repetitive and predictable than natural language," with identifier patterns as key statistical signals — a finding that anticipated the identifier dependence observed in modern code LMs.

### Code LMs and Identifier Dependence

CodeT5 (Wang et al., EMNLP 2021) designed identifier-aware pre-training objectives. Adversarial renaming attacks (Zhang et al., AAAI 2020; Bielik and Vechev, ICML 2020) and robustness studies (Rabin et al., IST 2021; Troshin and Chirkova, BlackboxNLP 2022) established that models exploit identifier semantics. Recent large-scale evaluations (Nikiema et al., 2025; Le et al., 2025) confirm the finding across contemporary model families including GPT-4o.

### Constrained Decoding and Structured Output

The constrained decoding literature — Outlines (Willard and Louf, TMLR 2023), LMQL (Beurer-Kellner et al., PLDI 2023), SGLang (Zheng et al., NeurIPS 2024), XGrammar (Dong et al., 2024), PICARD (Scholak et al., EMNLP 2021) — provides the mechanical enforcement of our structural channel. JSONSchemaBench (Geng et al., 2025) benchmarks compliance and quality. This literature grounds our "structural containment" argument: constrained decoding enforces $S_{\text{struct}}$ mechanically while leaving $S_{\text{lang}}$ to the neural consumer.

### Security and Prompt Injection

The prompt injection literature (Greshake et al., AISec@CCS 2023; Liu et al., USENIX Security 2024) and tool-poisoning work (Beurer-Kellner and Fischer, 2025; Wang et al., NAACL 2025; Wang et al., 2025) directly instantiate the security consequence of our thesis: if names are instructions, they are also attack vectors. These works are discussed in [§9](#9-security-implications-adversarial-indexing).

### Linguistic Relativity

The observation that schema vocabulary shapes the neural consumer's output distribution is structurally analogous to linguistic relativity (Whorf, 1956) — applied to neural rather than human cognition. We note this analogy without claiming mechanistic equivalence: what makes it useful is that schema-consuming systems provide a setting where the relevant experimental controls are tighter than those available in the human cognitive science literature. $S_{\text{struct}}$ can be held exactly constant while $S_{\text{lang}}$ is varied, and the output distribution can be measured precisely. SGD-X's crowdsourced paraphrases ([§4.1](#41-schema-guided-dialogue)), Dr.Spider's systematic perturbations ([§4.2](#42-text-to-sql-schema-linking)), and code obfuscation studies ([§4.3](#43-code-language-models-and-identifier-semantics)) are, in effect, controlled Whorfian experiments — not proof that neural consumers "think in language," but evidence that their output distributions are not invariant under the vocabulary of their input schemas.

---

## 13. Limitations

**Semantic equivalence of paraphrases.** SGD-X paraphrases are crowdsourced and designed to preserve meaning, but natural language does not admit perfect semantic identity. Our theoretical claim is best stated as "observational distinguishability under linguistically varied but intended-equivalent schema descriptions" rather than asserting perfect semantic identity of the varied descriptions.

**Mechanism may differ across domains.** The evidence supports convergent behavior — neural consumers leverage lexical semantics in schema identifiers across dialogue, SQL, and code domains — but we do not claim mechanistic identity. The same statistical phenomenon (sensitivity to naming) may arise from different learned representations in different model architectures and training regimes.

**Robustness is an open problem.** SGD-X demonstrates that schema paraphrases can degrade performance substantially. A practitioner relying on semantic indices faces the same fragility that SGD-X documents: small changes in wording can produce large changes in output. Progressive hardening ([§7](#7-the-precision-compression-design-space)) mitigates this by converting fragile semantic contracts into robust structural guarantees, but cannot eliminate the fundamental dependence on consumer capability for the semantic channel.

**Cross-channel interaction.** The two-channel model is a deliberate simplification. It treats the structural and semantic channels as independent constraint systems whose effects compose. In practice, the channels interact. Tam et al. (EMNLP 2024) found that strict constrained decoding can degrade reasoning quality while enhancing classification accuracy, suggesting that mechanical enforcement of $S_{\text{struct}}$ can distort the semantic processing that $S_{\text{lang}}$ relies on. Park et al. (NeurIPS 2024) proposed Grammar-Aligned Decoding specifically to address distribution distortion effects of naive grammar constraints. The two-channel decomposition remains useful as an analytical tool — it correctly predicts that structural guarantees hold while semantic behavior varies — but the channels are not fully independent, and a complete account of their interaction is an open problem.

**Metric with pending measurement.** We define semantic precision as a measurable quantity ($\Delta H_f$, [§7](#7-the-precision-compression-design-space)) and describe an experiment designed to measure it ([§8](#8-experiments-)). Until those measurements are reported, the information-theoretic framing is a predictive framework whose empirical validation is in progress rather than complete.

**Consumer capability is a moving target.** The magnitude of semantic indexing effects depends on the consumer's language understanding, which varies across model families and improves with scale. Effects documented today may attenuate or amplify as models evolve.

---

## 14. Conclusion

When a language model consumes a type schema, names stop being addresses and become instructions. This paper has defined the resulting phenomenon — semantic index types — formalized a measurable quantity that captures it ($\Delta H_f$), and situated both within a two-channel constraint system whose information-theoretic structure governs engineering, security, and the limits of semantic influence simultaneously.

The explanation is architectural. Traditional types compile to machine code that erases names. Schema-driven types compile to token sequences consumed by a neural network that reads them. Alpha equivalence fails because the compilation target changed. The two-channel model follows: the structural channel determines the support of the output distribution (mechanically enforced), the semantic channel determines the conditional probabilities within that support (neurally guided), and $I(N; Y_f) \leq \log_2 |V_f|$ — the semantic channel's influence is bounded by the structural compression of the type. This bound governs both the engineering design space and the security attack surface. Progressive hardening reduces semantic bandwidth, converting soft guidance into hard proof. An adversary who controls the semantic channel can influence at most as many bits as the structural constraint leaves open. The engineering story and the security story are the same story.

Three research communities — schema-guided dialogue, text-to-SQL, and code language models — independently discovered what amounts to linguistic relativity for neural consumers: the vocabulary of the schema determines the output distribution of the model. What this paper contributes is the recognition that these independently observed effects are instances of a single underlying property, a formal characterization of that property, and a metric that makes it measurable. We describe an experiment ([§8](#8-experiments-)) designed to measure $\Delta H_f$ across structural compression levels in Pydantic structured output — directly testing whether the information-theoretic prediction holds in the paper's target domain.

The theory of semantic index types is the theory of what happens when naming becomes programming.

---

## References

Barendregt, H.P. (1984). *The Lambda Calculus: Its Syntax and Semantics*. Revised edition. North-Holland.

Beurer-Kellner, L., Fischer, M., and Vechev, M. (2023). Prompting is programming: A query language for large language models. *PLDI 2023*.

Beurer-Kellner, L. and Fischer, M. (2025). Tool poisoning attacks on AI agents. Invariant Labs.

Bielik, P. and Vechev, M. (2020). Adversarial robustness for code. *ICML 2020*.

Chang, S., et al. (2023). Dr.Spider: A diagnostic evaluation benchmark towards text-to-SQL robustness. *ICLR 2023*.

Chen, B., et al. (2025). StruQ: Defending against prompt injection with structured queries. *USENIX Security 2025*.

Chen, Z., et al. (2021). ShadowGNN: Graph projection neural network for text-to-SQL parser. *NAACL 2021*.

Church, A. (1941). *The Calculi of Lambda Conversion*. Annals of Mathematics Studies, 6.

Dong, Y., et al. (2024). XGrammar: Flexible and efficient structured generation engine for large language models. Preprint.

Geng, S., et al. (2025). JSONSchemaBench: A benchmark for structured generation with complex JSON schemas. Preprint.

Greshake, K., et al. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *AISec@CCS 2023*.

Hindle, A., et al. (2012). On the naturalness of software. *ICSE 2012*. (Most Influential Paper, 2022.)

King, A. (2019). Parse, don't validate. Blog post.

Le, H., et al. (2025). When names disappear: Benchmarking LLMs for code generation without natural language cues. Preprint.

Lee, H., et al. (2022). SGD-X: A benchmark for robust generalization in schema-guided dialogue systems. *AAAI 2022*.

Lei, W., et al. (2020). Re-examining the role of schema linking in text-to-SQL. *EMNLP 2020*.

Li, J., et al. (2023). Can LLM already serve as a database interface? A BIg bench for large-scale database grounded text-to-SQL (BIRD). *NeurIPS 2023*.

Lin, X.V., et al. (2020). Bridging textual and tabular data for cross-domain text-to-SQL semantic parsing. *EMNLP 2020*.

Liu, Y., et al. (2024). Formalizing and benchmarking prompt injection attacks and defenses. *USENIX Security 2024*.

Nikiema, P., et al. (2025). The code barrier: What LLMs actually understand? Preprint.

Park, K., et al. (2024). Grammar-aligned decoding. *NeurIPS 2024*.

Rabin, M.R.I., et al. (2021). Understanding neural code intelligence through program simplification. *IST 2021*.

Rastogi, A., et al. (2020). Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset. *AAAI 2020*.

Scholak, T., et al. (2021). PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. *EMNLP 2021*.

Tam, Z.R., et al. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. *EMNLP 2024*.

Troshin, S. and Chirkova, N. (2022). Probing pretrained models of source code. *BlackboxNLP@EMNLP 2022*.

Wallace, E., et al. (2024). The instruction hierarchy: Training LLMs to prioritize privileged instructions. OpenAI.

Wang, B., et al. (2020). RAT-SQL: Relation-aware schema encoding and linking for text-to-SQL parsers. *ACL 2020*.

Wang, Y., et al. (2021). CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. *EMNLP 2021*.

Wang, Z., et al. (2025). MCPTox: A broad evaluation of LLM agent safety through tool poisoning. Preprint.

Wang, Z., et al. (2025). ToolCommander: Adversarial attacks and defenses in multi-turn LLM tool-use. *NAACL 2025*.

Whorf, B.L. (1956). *Language, Thought, and Reality: Selected Writings of Benjamin Lee Whorf*. Edited by J.B. Carroll. MIT Press.

Willard, B. and Louf, R. (2023). Efficient guided generation for large language models. *TMLR 2023*.

Wretblad, N., et al. (2024). Understanding the effects of column descriptions on text-to-SQL. *NeurIPS 2024*.

Zhang, H., et al. (2020). Generating adversarial examples for holding robustness of source code processing models. *AAAI 2020*.

Zhao, J., et al. (2022). Description-driven task-oriented dialog modeling. Preprint.

Zheng, L., et al. (2024). SGLang: Efficient execution of structured language model programs. *NeurIPS 2024*.
