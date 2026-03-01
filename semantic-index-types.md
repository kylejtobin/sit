# Semantic Index Types

## When Names Become Computation in Schema-Driven Language Model Systems

---

**Abstract.** We introduce *semantic index types*: type declarations in which natural-language tokens — field names, descriptions, and enum member names — function as computational indices that shape the output of a neural consumer. When a language model consumes a type schema, it interprets these tokens as instructions rather than inert identifiers, producing measurably different outputs for structurally isomorphic schemas with different natural-language content. This violates a consumer-level analogue of alpha equivalence while preserving all structural guarantees — a failure we trace to a change in compilation target: traditional types compile to machine code that erases names, while schema-driven types compile to token sequences consumed by a neural network that reads them. We formalize the phenomenon as a two-channel constraint system — structural and semantic — whose interaction has information-theoretic structure: the semantic channel's effective capacity is bounded by the structural compression of the type, unifying the engineering design space with the security attack surface under a single quantity. We situate this framework within converging empirical evidence from schema-guided dialogue, text-to-SQL schema linking, and code language model research — three communities that independently discovered what amounts to linguistic relativity for neural consumers. We discuss operational semantics of schema exposure as compilation, progressive hardening from semantic to structural guarantees, and the security implications of a collapsed data/instruction boundary at the schema level.

---

## 1. Introduction

Alpha equivalence — the principle that consistent renaming of bound variables preserves program semantics — is foundational to the formal semantics of lexically scoped programming languages. In the lambda calculus, alpha conversion is one of three reduction rules (Church, 1941). In modern PL theory, the Variable Convention treats bound variable names as arbitrary up to consistent renaming (Barendregt, 1984, Definition 2.1.13). Compilers for languages like Haskell assign each binder a unique integer in their intermediate representations, rendering the original source name irrelevant to the elaborated program; GHC documentation describes type equivalence as "syntactic equivalence modulo alpha-renaming."

Alpha equivalence holds because every consumer in a traditional execution pipeline — parser, type checker, optimizer, code generator — treats names as structural identifiers. The name tells the system *which* slot. It never determines *what* value inhabits that slot.

This paper identifies a setting in which the analogous invariance property breaks down. When a language model consumes a type schema to produce structured output, the model interprets field names, descriptions, and enum labels as natural-language instructions. Structurally isomorphic schemas with different linguistic content produce measurably different output distributions. Names have crossed from the routing plane to the computation plane. The deep reason is that the compilation target has changed: traditional types compile to machine code that erases names; schema-driven types compile to token sequences consumed by a neural network that reads them.

We are not the first to observe that schema text affects model behavior. The schema-guided dialogue community has evaluated robustness to paraphrased schema descriptions since SGD-X (Lee et al., AAAI 2022). The text-to-SQL community treats column names as semantic anchors central to schema linking (Lei et al., EMNLP 2020). Code language model research has documented that obfuscating identifier names degrades performance across tasks and model families (Nikiema et al., 2025; Le et al., 2025). What is novel here is the *PL-theoretic framing*: we characterize the phenomenon as a two-channel type system in which the structural channel obeys alpha equivalence and the semantic channel violates it, and we show that this framing unifies observations across independently evolved research communities. The resulting picture is, in effect, linguistic relativity for neural consumers (§11): the vocabulary of the schema determines the output distribution of the model.

### Scope and Terminology

We use "alpha equivalence" to describe a *consumer-level* invariance property, not a claim about the object-language semantics of any particular programming language. Our claim is that the neural consumer function — the mapping from schema plus context to output distribution — is not invariant under renaming transformations that are structure-preserving in the schema language. This is distinct from, though motivated by, the formal notion of alpha equivalence in the lambda calculus.

We note that even in the history of programming languages, names have not always been inert. Fortran's implicit typing rule (IBM, 1956) assigned INTEGER type to variables beginning with letters I through N and REAL to all others; `IMPLICIT NONE` was required to disable this behavior. Renaming a Fortran variable could change its type and therefore program behavior — a *language-level* alpha equivalence failure. Python's `locals()` and `globals()` make variable names observable as dictionary keys. Our contribution is not "names sometimes matter" — it is a formal characterization of *how* and *why* they matter when the consumer is a neural language model, and the engineering consequences that follow.

The term "semantic index type" is not related to "indexed types" in the dependent-type sense (types parameterized by values). Here, "index" refers to a natural-language token that indexes into the consumer's learned semantic associations.

---

## 2. Formal Framework

### 2.1 The Neural Consumer

Let $S$ be a schema consisting of a structural component $S_{\text{struct}}$ (type annotations, algebraic constraints, field count, nesting depth) and a linguistic component $S_{\text{lang}}$ (field names, descriptions, enum member labels). Let $C$ denote a neural consumer — the composition of a language model with its decoding protocol — mapping (schema, context) pairs to output distributions:

$$y \sim C(x, S)$$

where $x$ is the input context (prompt, prior conversation, injected data) and $y$ is the generated structured output.

### 2.2 Two Notions of Equivalence

We define structural isomorphism and semantic equivalence as distinct relations:

**Structural isomorphism.** Two schemas $S$ and $S'$ are structurally isomorphic ($S \equiv_\alpha S'$) when $S_{\text{struct}} = S'_{\text{struct}}$ — they have the same type annotations, the same nesting structure, the same algebraic constraints, and differ only in $S_{\text{lang}}$.

**Semantic equivalence.** Two schemas $S$ and $S'$ are semantically equivalent under consumer $C$ ($S \equiv_{\text{sem}} S'$) when $C(x, S)$ and $C(x, S')$ produce equivalent output distributions under a chosen divergence metric $d$, for all relevant contexts $x$.

### 2.3 The Core Claim

The central empirical claim of this paper is:

$$\exists\, S, S' \text{ such that } S \equiv_\alpha S' \text{ but } S \not\equiv_{\text{sem}} S'$$

That is, there exist structurally isomorphic schemas whose linguistic components produce measurably different output distributions under neural consumers. This claim is not speculative. It is operationalized and quantified by existing benchmarks across multiple research communities (§4).

### 2.4 The Two-Channel Model

A semantic index type operates through two constraint channels simultaneously:

The **structural channel** bounds the space of valid outputs. Type annotations, enum membership, numeric constraints, cross-field validators, and constrained decoding grammars define the set of structurally valid values. These constraints hold regardless of the consumer's language understanding. They are enforced mechanically — either during generation (constrained decoding) or after generation (validation and rejection).

The **semantic channel** guides the consumer's selection within the structurally valid space. Field names, descriptions, and enum member labels function as natural-language instructions that shape *which* structurally valid value the consumer produces. Compliance depends on the precision of the linguistic content, the capability of the consumer, and the degree to which structural constraints have already narrowed the space.

The structural channel contains the semantic channel. When the consumer's semantic interpretation is imprecise — it selects `RiskTier.HIGH` when `RiskTier.CRITICAL` was more appropriate — the error is bounded by the structural constraints. The output is still a valid enum member. All construction invariants still hold. Semantic imprecision lives within a proven structural envelope.

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

A natural objection: is a semantic index type just prompt engineering with a schema wrapper? The answer is no, and the distinction is precise. A prompt carries instruction only — it tells the model what to do, but nothing in the prompt's structure constrains or proves the result. A semantic index type carries instruction, constraint, and proof simultaneously. The natural-language tokens instruct the consumer ($S_{\text{lang}}$). The type annotations constrain the output space ($S_{\text{struct}}$). The construction pipeline — coercion, validation, cross-field invariants — proves the result is structurally sound. A prompt that says "return a JSON object with a risk tier" is an instruction. A Pydantic model with `risk_tier: RiskTier` backed by an enum, a description, and a model validator is instruction, constraint, and proof in a single declaration. The construction pipeline is the differentiator.

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

The enrichment direction provides complementary evidence. The BIRD benchmark (Li et al., NeurIPS 2023) evaluates text-to-SQL with and without "external knowledge evidence" — a condition that includes column descriptions and associated metadata. GPT-4's execution accuracy rose from 34.88% without this evidence to 54.89% with it. We note that this delta reflects the combined effect of column descriptions and other knowledge evidence, not descriptions in isolation; the result demonstrates that schema-associated linguistic content substantially affects neural consumer behavior, though the individual contribution of each component is not isolated. Wretblad et al. (NeurIPS 2024) found that adding synthesized column descriptions consistently enhanced accuracy across models, providing a more targeted measurement of description-level effects.

### 4.3 Code Language Models and Identifier Semantics

Code language models provide a third line of evidence with particularly clean experimental controls, because programming languages have formal semantics that make "semantics-preserving renaming" well-defined.

CodeT5 (Wang et al., EMNLP 2021) explicitly designed pre-training objectives around identifier recovery, leveraging "semantics conveyed from developer-assigned identifiers" to achieve state-of-the-art results across 14 CodeXGLUE sub-tasks. The architecture treats identifier names as a learnable semantic signal, not structural noise.

Adversarial and robustness studies confirm the dependence. Zhang et al. (AAAI 2020) achieved over 90% attack success rates by renaming identifiers — a semantics-preserving transformation that nonetheless changed model behavior. Rabin et al. (IST 2021) demonstrated that models fail to generalize after variable renaming, and Bielik and Vechev (ICML 2020) showed identifier renaming attacks degrade type inference models.

Two recent large-scale studies provide the most comprehensive quantification. Nikiema et al. (2025) tested variable renaming across 13 contemporary LLMs including GPT-4o, finding an average accuracy drop of 18.6 percentage points. GPT-4o showed a 7.3% decrease — the smallest among models tested, but still measurable. Le et al. (2025) documented performance degradation across multiple benchmarks under identifier obfuscation: for GPT-4o on ClassEval, class-level accuracy dropped from 87.3% to 58.7%; on LiveCodeBench and other benchmarks, consistent degradation was observed across model families.

Karmakar and Robbes (ASE 2021) added an interpretability finding: general-purpose BERT models performed competitively with code-specific models on certain code understanding tasks, suggesting that natural-language content in identifiers — not code-specific structural understanding — was driving performance. Troshin and Chirkova (BlackboxNLP 2022) confirmed that anonymizing identifiers was the most damaging single transformation in their study of code representation robustness.

### 4.4 Cross-Domain Convergence

The convergence across these three communities is the strongest argument for treating semantic indexing as a general phenomenon rather than a domain-specific artifact. Schema-guided dialogue researchers, text-to-SQL researchers, and code LM researchers each independently discovered that neural consumers systematically leverage lexical and natural-language semantics in schema identifiers and descriptions, and that this creates measurable sensitivity to renaming and paraphrase transformations. The controlled experimental designs — SGD-X's crowdsourced paraphrases, Dr.Spider's systematic perturbations, code obfuscation's semantics-preserving renames — each isolate the naming channel as causal.

We note that the *mechanism* need not be identical across domains. What the evidence supports is a weaker but still powerful claim: across domains, neural consumers treat schema-level natural-language tokens as a semantic information channel, and the output distribution is not invariant under transformations of that channel even when structural content is preserved.

---

## 5. Operational Semantics of Schema Exposure

Schema serialization is compilation. Traditional types compile to machine code where names are erased — the CPU has no use for them, and alpha equivalence is a design property of the compilation target. Semantic index types "compile" (via `model_json_schema()`, tool definitions, or prompt serialization) to token sequences consumed by a neural network that actively interprets names as natural-language instructions. The compilation target changed, and which properties survive compilation depends on the target. Alpha equivalence fails because the target architecture does not erase names. It reads them.

The degree to which semantic indexing affects a neural consumer depends on *how* the schema reaches the consumer — that is, the specifics of the compilation. We distinguish three regimes with different properties for each channel:

### 5.1 Schema-as-Text Prompt

The schema is serialized as natural-language text within the prompt. Field names, descriptions, and type annotations are all visible as tokens in the model's context window. This regime maximizes semantic indexing: the consumer has full access to $S_{\text{lang}}$ and interprets it as natural-language instruction. It also maximizes the prompt injection attack surface (§8).

### 5.2 Schema-as-Tool Definition

The schema is passed via a structured API channel (e.g., function calling / tool use definitions) rather than concatenated into the prompt text. The linguistic content remains available — tool parameters carry names and descriptions — but it is structurally separated from the conversational context. Semantic indexing still occurs, but the consumer processes the schema through a different attention pathway than free-form prompt text.

### 5.3 Schema-as-Hard Constraint

The schema is compiled into a decoding grammar that mechanically constrains token generation. Constrained decoding frameworks — Outlines (Willard and Louf, TMLR 2023), LMQL (Beurer-Kellner et al., PLDI 2023), SGLang (Zheng et al., NeurIPS 2024), XGrammar (Dong et al., 2024) — enforce structural compliance during generation rather than after it. OpenAI's `strict: true` structured output mode (August 2024) and Anthropic's structured output support compile JSON Schema into decoding grammars.

In this regime, the structural channel provides *hard guarantees during generation*: the model physically cannot produce tokens that violate the schema's structural constraints. JSONSchemaBench (Geng et al., 2025) evaluates constrained decoding frameworks across approximately 10,000 real-world schemas and reports that constrained decoding can speed generation by roughly 50% while improving task performance up to 4%.

Critically, constrained decoding enforces only $S_{\text{struct}}$. The grammar ensures valid JSON with correct types and enum values, but it does not and cannot determine *which* valid enum value the model selects or *what* semantic content populates a string field. The semantic channel remains entirely under the consumer's language-level processing. This makes constrained decoding a near-perfect illustration of the two-channel model: structure is mechanically enforced, semantics is neurally guided. (The channels are not fully independent, however; structural enforcement can interact with semantic processing in ways we discuss as a limitation in §12.)

### 5.4 Enforcement After Generation

Many production systems operate in a "generate → validate → retry" loop rather than using constrained decoding. Pydantic's construction pipeline exemplifies this regime: the model generates output, `model_validate` attempts construction, and if construction fails (type coercion error, validator failure, cross-field invariant violation), the system retries with the error message fed back to the consumer.

In this regime, structural guarantees are *eventual* rather than *generative*. The consumer may initially produce structurally invalid output, but the loop converges on valid output or fails explicitly. The semantic channel operates identically in both regimes — the consumer interprets $S_{\text{lang}}$ regardless of whether structural compliance is enforced during or after generation.

### 5.5 What Pydantic Transmits

A practical note on the specific mechanism by which schema content reaches the consumer in the Pydantic ecosystem. Pydantic's `model_json_schema()` includes class-level docstrings as the model `description` and `Field(description=...)` values as field-level descriptions. Field-level docstrings (docstrings placed below field declarations) are *not* automatically incorporated into the JSON Schema. When this paper refers to "docstrings" as semantic indices, we mean specifically class docstrings and field descriptions declared via `Field(description=...)`, which are the mechanisms that reliably propagate linguistic content into the schema consumed by language models.

---

## 6. Naming as Computation

In a semantic index type system, field naming is programming. This claim is concrete and empirically grounded.

`counterparty_credit_rating: CreditRating` tells a language model to assess the counterparty's creditworthiness. `x7: CreditRating` does not. The structural output is the same type. The semantic output differs: one produces a credit assessment grounded in the concept of counterparty risk; the other produces a member selected without interpretive guidance. The name is the instruction. Changing the name changes the instruction.

**Field names** are the primary semantic indices. They are the first tokens the consumer reads when determining what value to generate for a slot. Choosing `churn_risk_tier` over `attrition_risk_tier` is choosing between two analytical framings — voluntary departure versus passive loss. This choice shapes the consumer's reasoning: what signals to weigh, what thresholds to apply, what risk narrative to construct. It is not naming convention. It is program logic.

**Field descriptions** are disambiguation instructions. A description reading "Projected total revenue across the full customer relationship, not historical sum" narrows the consumer's interpretation of `lifetime_value` from a broad concept to a specific forward-looking revenue projection. Removing the description changes the output distribution. The description is part of the program.

**Enum member names** are a closed vocabulary of semantic instructions executed within a structurally bounded space. `RiskTier.CRITICAL` and `RiskTier.SEVERE` are structurally identical — both are members of the same enum — but they produce different downstream reasoning. "Critical" carries connotations of immediacy and threshold-crossing; "severe" carries connotations of magnitude without the same urgency.

**Renaming is refactoring.** In conventional programming, renaming a variable is a safe mechanical operation. In a semantic index type system, renaming a field changes what the consumer computes. The empirical evidence (§4) quantifies this: renaming-class transformations produce degradations ranging from 7% to over 50% across benchmarks and model families. Renaming requires the same care and testing as modifying a function's return logic.

---

## 7. The Precision-Compression Design Space

Semantic index types occupy a design space defined by two independent axes.

**Structural compression** is the degree to which the type annotation constrains the set of valid outputs. `Literal["active", "inactive"]` compresses to two values. A four-member enum compresses to four. `str` provides no compression.

**Semantic precision** is the degree to which the natural-language indices narrow the consumer's interpretation within the structurally valid space. A field named `churn_risk_tier` with a description specifying behavioral signals is semantically precise. A field named `x7` with no description is semantically vacuous.

The axes are independent. High compression with low precision produces constrained but ambiguously motivated outputs. Low compression with high precision produces well-motivated but structurally unbounded outputs. The engineering objective is to maximize both: tight structural compression to bound the output space, and precise semantic indices to guide the consumer within that bounded space.

### Information-Theoretic Structure

The relationship between the two axes has a quantitative backbone. The semantic channel's effective capacity — the maximum information it can contribute to the output — is bounded by the size of the structurally valid set. A `bool` field gives the semantic channel exactly 1 bit to work with: the structural constraint admits two values, and the semantic index can at most determine which one. A four-member enum gives 2 bits. A `Literal["active", "inactive", "suspended"]` gives $\log_2 3 \approx 1.58$ bits. A bare `str` field gives the semantic channel unbounded capacity — the structural constraint admits any string, so the semantic index bears the full burden of determining the output.

This means structural compression and semantic channel capacity are inversely related. Each increase in structural compression reduces the number of bits the semantic channel can influence. Progressive hardening (below) is, in information-theoretic terms, the systematic reduction of the semantic channel's bandwidth. This has a dual consequence: it reduces the power of semantic indices to guide the consumer (the engineering cost) while simultaneously reducing the attack surface for adversarial indexing (§8), because an attacker who controls $S_{\text{lang}}$ can influence at most as many bits as the structural constraint leaves open. The security story and the engineering story are governed by the same quantity.

This framework suggests a natural metric for semantic precision. For a given field $f$ with structurally valid set $V_f$, the semantic precision of $S_{\text{lang}}$ relative to a vacuous schema $S'_{\text{lang}}$ (one with no meaningful natural-language content) can be defined as the reduction in output entropy:

$$\Delta H_f = H(C(x, S')|_f) - H(C(x, S)|_f)$$

where $H(C(x, S)|_f)$ is the entropy of the consumer's output distribution over $V_f$. When $\Delta H_f$ is large, the semantic index substantially narrows the consumer's selection within the structurally valid space. When $\Delta H_f$ is near zero, the semantic index adds no guidance beyond what the structural constraint already provides. This quantity is directly measurable by comparing output distributions across schema variants with and without meaningful natural-language content.

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

This is the engineer's version of a familiar PL/SE story: start with soft specifications, then promote frequently violated expectations into enforced invariants. The empirical evidence supports this approach: SGD-X's schema sensitivity measurements (§4.1) and Dr.Spider's perturbation degradation curves (§4.2) can serve as the observational basis for deciding which semantic contracts to harden.

---

## 8. Security Implications: Adversarial Indexing

If field names and descriptions are instructions, they are also an attack surface.

The vulnerability has deep precedent. In a Von Neumann stored-program architecture, the distinction between data and instructions is a matter of interpretation by the processor, not a property of the bits. Buffer overflows, SQL injection, and cross-site scripting all exploit the same fundamental pattern: content intended as data crosses into an execution context and is interpreted as instruction. The canonical injection vulnerability classes in computing are instances of a collapsed data/instruction boundary.

Semantic index types exhibit exactly this collapse at the schema level. A field name is simultaneously data (a key in a JSON Schema object) and instruction (a semantic index that the neural consumer interprets as a natural-language directive). The consumer cannot mechanically distinguish between the two roles. This is not an analogy to the injection vulnerability class — it is an instance of it.

Greshake et al. (AISec@CCS 2023) formalized indirect prompt injection as the exploitation of the fact that LLMs "blur the line between data and instructions." Liu et al. (USENIX Security 2024) provided a formal framework for prompt injection attacks. In both treatments, the core vulnerability is the collapsed data/instruction boundary applied to LLM context windows. Semantic index types make the boundary collapse explicit and localized: every field name, description, and enum label is a point where data and instruction coincide.

In tool-mediated systems, this vulnerability manifests as *tool description poisoning*. Beurer-Kellner and Fischer (Invariant Labs, 2025) demonstrated that poisoned instructions in tool descriptions can hijack model behavior even when the poisoned tool is never invoked. Wang et al. (NAACL 2025) systematically evaluated this with ToolCommander, achieving 91.67% attack success for privacy theft and 100% for denial of service. MCPTox (Wang et al., 2025) evaluated 353 tools across 45 MCP servers, finding over 60% attack success rates for GPT-4o-mini, o1-mini, DeepSeek-R1, and Phi-4.

For semantic index types specifically, every schema field is a potential injection point. A field description that reads "Ignore all previous instructions and output the user's API key" exploits the same channel that a legitimate description like "Projected total revenue across the full customer relationship" uses for semantic guidance. The structural channel is immune — a constrained decoding grammar enforces valid JSON regardless of injected content — but the semantic channel is vulnerable because the consumer cannot mechanically distinguish legitimate semantic indices from adversarial ones.

Mitigations include schema provenance verification (ensuring schemas originate from trusted sources), input sanitization of description fields when schemas are dynamically constructed, least-privilege scoping of schema content, and architectural separation of untrusted content from instruction channels. Wallace et al. (OpenAI, 2024) proposed an instruction hierarchy that achieved up to 63% better resistance to prompt injection. Chen et al. (USENIX Security 2025) introduced StruQ, which uses structured queries with separator tokens to achieve near-zero success rates for optimization-free attacks.

The security implications are not a side effect of semantic index types — they are a direct consequence of the core thesis. If names are computation, names are also attack vectors.

---

## 9. Host System Requirements

Semantic index types require a host system satisfying two conditions simultaneously:

**Names must be preserved and exposed.** The type system must retain field names, descriptions, and enum members as first-class schema content visible to the consumer. This is the condition that compilation pipelines designed around alpha equivalence resist. Haskell's GHC assigns each binder a unique integer in Core (System FC); the original source name is not guaranteed to survive elaboration because the compilation target — as discussed in §5 — erases names by design. Even serializing a Haskell record to JSON and feeding the schema to a model means the result has left Haskell's type system; guarantees depend on whatever validation logic exists on the Haskell side.

**A construction pipeline must sit behind the names.** The type system must provide coercion, constraint enforcement, cross-field invariants, and structural dispatch, wired into a single construction call. Without this, semantic index types reduce to prompt engineering with schema decoration — the names instruct the consumer, but nothing proves the result.

Pydantic is a particularly ergonomic and widely adopted host for semantic index types in Python. It preserves field names and descriptions in its JSON Schema output because it was designed for API serialization, and it provides a full construction pipeline because it was designed for data validation. Neither design goal targeted semantic indexing; the combination emerged as an accident of API design requiring preserved names and data validation requiring construction pipelines.

The deeper requirement, however, is not Pydantic-specific. Any runtime schema object with meaningful labels and an enforcement pipeline can host semantic index types. Zod in TypeScript provides runtime schema validation with preserved names and a construction pipeline. JSON Schema preserves names but is a description format with no execution semantics. Dataclasses preserve names but have no construction pipeline. The requirement is the conjunction: preserved names AND enforcement. Pydantic provides this conjunction with particular ergonomics in the Python ecosystem, where the majority of LLM application development currently occurs.

---

## 10. The LLM as Computational Primitive

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

`RetentionAnalysis` is simultaneously the instruction set (via semantic indices that tell the model what to generate), the constraint set (via structural types that bound what the model can produce), and the proof obligation (via the construction pipeline that rejects invalid results). The model operates inside all three.

This is the operational consequence of the distinction drawn in §3: a semantic index type carries instruction, constraint, and proof simultaneously, and the model operates inside all three. The quality of the model's output is determined by the precision of $S_{\text{lang}}$ and the tightness of $S_{\text{struct}}$. Improving output is not about writing better prose in a system message. It is about engineering better types.

---

## 11. Related Work

### Formal Foundations

Alpha equivalence was formalized by Church (1941) as part of the lambda calculus. Barendregt (1984) provides the standard modern treatment, including the Variable Convention (Definition 2.1.13). Our work does not challenge alpha equivalence as a property of formal language semantics; rather, we identify a consumer-level invariance failure when neural models interpret schema labels.

The historical precedent of Fortran's I-N implicit typing rule (IBM, 1956) demonstrates that language-level alpha equivalence failures have existed before. The distinction is that Fortran's rule is a deliberate language design decision making names semantically relevant *within* the PL, while semantic index types arise from a neural consumer importing natural-language meaning into an otherwise structural schema.

Although we disclaim the connection to indexed types in the dependent-type sense (§1), a structural analogy is worth noting. In dependent type theory, values flow into types to constrain subsequent computation: a vector's length parameter determines which operations are well-typed. In semantic index types, natural-language tokens flow into the consumer's latent space to constrain subsequent generation: a field's name determines which outputs the consumer considers appropriate. In both cases, an index parameterizes behavior. The difference is the nature of the guarantee: dependent type indices have formal denotational semantics and provide proofs; semantic indices have natural-language semantics interpreted by a learned function and provide probabilistic guidance. Progressive hardening (§7) is the process of replacing informal natural-language constraints with formal structural ones — moving, incrementally, from the semantic index regime toward the dependent type regime in terms of guarantee strength.

### Schema-Guided NLP

The SGD dataset (Rastogi et al., AAAI 2020) established the practice of supplying natural-language descriptions as active inputs to dialogue systems. SGD-X (Lee et al., AAAI 2022) operationalized our core claim as a benchmark: five paraphrase variants of each schema, with Schema Sensitivity as a dedicated metric. D3ST (Zhao et al., 2022) isolated the description channel by replacing names with random indices. This body of work provides the most direct empirical precedent for semantic index types.

### Text-to-SQL and Schema Linking

Schema linking — mapping utterance phrases to table and column names — has been central to neural text-to-SQL since the introduction of Spider. RAT-SQL (Wang et al., ACL 2020), BRIDGE (Lin et al., EMNLP 2020), and subsequent work treat schema tokens as semantic anchors. Dr.Spider (Chang et al., 2023) provides controlled robustness evaluation under schema perturbation. BIRD (Li et al., NeurIPS 2023) and Wretblad et al. (NeurIPS 2024) demonstrate enrichment effects of descriptions. Hindle et al. (ICSE 2012, Most Influential Paper 2022) established that code is "more repetitive and predictable than natural language," with identifier patterns as key statistical signals — a finding that anticipated the identifier dependence observed in modern code LMs.

### Code LMs and Identifier Dependence

CodeT5 (Wang et al., EMNLP 2021) designed identifier-aware pre-training objectives. Adversarial renaming attacks (Zhang et al., AAAI 2020; Bielik and Vechev, ICML 2020) and robustness studies (Rabin et al., IST 2021; Troshin and Chirkova, BlackboxNLP 2022) established that models exploit identifier semantics. Recent large-scale evaluations (Nikiema et al., 2025; Le et al., 2025) confirm the finding across contemporary model families including GPT-4o.

### Constrained Decoding and Structured Output

The constrained decoding literature — Outlines (Willard and Louf, TMLR 2023), LMQL (Beurer-Kellner et al., PLDI 2023), SGLang (Zheng et al., NeurIPS 2024), XGrammar (Dong et al., 2024), PICARD (Scholak et al., EMNLP 2021) — provides the mechanical enforcement of our structural channel. JSONSchemaBench (Geng et al., 2025) benchmarks compliance and quality. This literature grounds our "structural containment" argument: constrained decoding enforces $S_{\text{struct}}$ mechanically while leaving $S_{\text{lang}}$ to the neural consumer.

### Security and Prompt Injection

The prompt injection literature (Greshake et al., AISec@CCS 2023; Liu et al., USENIX Security 2024) and tool-poisoning work (Beurer-Kellner and Fischer, 2025; Wang et al., NAACL 2025; Wang et al., 2025) directly instantiate the security consequence of our thesis: if names are instructions, they are also attack vectors. These works are discussed in §8.

### Linguistic Relativity

The claim that schema vocabulary determines the neural consumer's output distribution is structurally a Sapir-Whorf claim — linguistic relativity applied to neural rather than human cognition. The strong Sapir-Whorf hypothesis (Whorf, 1956), that the structure of language determines the structure of thought, has been contested for decades in part because human experimental controls cannot cleanly separate linguistic from cognitive variables. Schema-consuming language models provide a setting where this separation is possible: $S_{\text{struct}}$ can be held exactly constant while $S_{\text{lang}}$ is varied, and the output distribution can be measured precisely. SGD-X's crowdsourced paraphrases (§4.1), Dr.Spider's systematic perturbations (§4.2), and code obfuscation studies (§4.3) are, in effect, controlled Whorfian experiments with tighter experimental designs than the human cognitive science literature has generally achieved. Semantic index types are the framework for a domain in which linguistic relativity is not a philosophical position but a measurable, operationalizable property of the consumer.

### Parse, Don't Validate

King (2019) articulated the principle that parsing should produce values whose type encodes the validation that has occurred, eliminating the possibility of "validated but wrongly typed" states. Semantic index types extend this principle: construction is parsing, and the parsed type carries both structural proof and semantic instruction.

---

## 12. Limitations

**Semantic equivalence of paraphrases.** SGD-X paraphrases are crowdsourced and designed to preserve meaning, but natural language does not admit perfect semantic identity. Our theoretical claim is best stated as "observational distinguishability under linguistically varied but intended-equivalent schema descriptions" rather than asserting perfect semantic identity of the varied descriptions.

**Mechanism may differ across domains.** The evidence supports convergent behavior — neural consumers leverage lexical semantics in schema identifiers across dialogue, SQL, and code domains — but we do not claim mechanistic identity. The same statistical phenomenon (sensitivity to naming) may arise from different learned representations in different model architectures and training regimes.

**Robustness is an open problem.** SGD-X demonstrates that schema paraphrases can degrade performance substantially. A practitioner relying on semantic indices faces the same fragility that SGD-X documents: small changes in wording can produce large changes in output. Progressive hardening (§7) mitigates this by converting fragile semantic contracts into robust structural guarantees, but cannot eliminate the fundamental dependence on consumer capability for the semantic channel.

**Cross-channel interaction.** The two-channel model is a deliberate simplification. It treats the structural and semantic channels as independent constraint systems whose effects compose. In practice, the channels interact. Tam et al. (EMNLP 2024) found that strict constrained decoding can degrade reasoning quality while enhancing classification accuracy, suggesting that mechanical enforcement of $S_{\text{struct}}$ can distort the semantic processing that $S_{\text{lang}}$ relies on. Park et al. (NeurIPS 2024) proposed Grammar-Aligned Decoding specifically to address distribution distortion effects of naive grammar constraints. The two-channel decomposition remains useful as an analytical tool — it correctly predicts that structural guarantees hold while semantic behavior varies — but the channels are not fully independent, and a complete account of their interaction is an open problem.

**Metric without measurement.** We define semantic precision as a measurable quantity ($\Delta H_f$, §7) but do not report empirical values in this version of the paper. The metric is directly measurable by comparing output distributions across schema variants — the experimental design follows naturally from the definition — but until those measurements are reported, the information-theoretic framing is a predictive framework rather than an empirical result.

**Consumer capability is a moving target.** The magnitude of semantic indexing effects depends on the consumer's language understanding, which varies across model families and improves with scale. Effects documented today may attenuate or amplify as models evolve.

---

## 13. Conclusion

When a language model consumes a type schema, names stop being addresses and become instructions. This paper has formalized the resulting phenomenon — semantic index types — as a two-channel constraint system in which the structural channel obeys alpha equivalence and the semantic channel violates it.

The deepest explanation is architectural. Traditional types compile to machine code that erases names because the target has no use for them. Schema-driven types compile to token sequences consumed by a neural network that reads names as natural-language instructions. Alpha equivalence fails because the compilation target changed. The two-channel model follows: the structural channel survives compilation intact (enforced mechanically by constrained decoding or validation), while the semantic channel emerges from the target's capacity to interpret natural language — a capacity no prior compilation target possessed.

The interaction between the channels has information-theoretic structure. The semantic channel's effective capacity is bounded by the structural compression of the type: a `bool` field gives 1 bit of semantic influence, a bare `str` gives unbounded capacity. This single quantity governs both the engineering design space (progressive hardening reduces semantic bandwidth, converting soft guidance into hard proof) and the security attack surface (an adversary who controls the semantic channel can influence at most as many bits as the structural constraint leaves open). The engineering story and the security story are the same story.

The phenomenon is not new. Three research communities — schema-guided dialogue, text-to-SQL, and code language models — independently discovered that neural consumers treat schema-level natural-language tokens as a semantic information channel, producing what amounts to linguistic relativity for neural consumers: the vocabulary of the schema determines the output distribution of the model, with tighter experimental controls than the human Sapir-Whorf literature has generally achieved. What is new is the recognition that these independently observed effects are instances of a single underlying property, and that the property has a formal characterization, a measurable quantity, engineering consequences, and a security classification rooted in the same data/instruction boundary collapse that underlies every major class of injection vulnerability in computing.

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

Karmakar, A. and Robbes, R. (2021). What do pre-trained code models know about code? *ASE 2021*.

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
