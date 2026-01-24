# Paper Selection Criteria

This document formalizes the criteria for including papers in "The Foundations of Large Language Models, 1943-2025."

## Primary Criteria

Papers must satisfy **at least 2** of the following primary criteria for inclusion in the main body (Parts I-VII):

| Criterion | Definition | Examples |
|-----------|------------|----------|
| **Foundational Impact** | Introduced a concept or technique that became essential infrastructure for subsequent work | Backpropagation, LSTM, Transformer, Attention mechanism |
| **Paradigm Shift** | Changed how researchers approach a class of problems | Pretraining (BERT/GPT), RLHF, Scaling Laws |
| **Universal Adoption** | Technique is now standard across most frontier models | LayerNorm, Adam optimizer, RoPE, BPE tokenization |
| **Enabling Breakthrough** | Made previously impractical approaches feasible | FlashAttention, LoRA, Dropout, GPU training |
| **Revealed Emergence** | Discovered unexpected capabilities arising from scale or training | GPT-3 few-shot learning, Chain-of-Thought reasoning |

## Secondary Criteria

These strengthen the case for inclusion but are not sufficient on their own:

| Criterion | Definition |
|-----------|------------|
| **Historical Significance** | First to demonstrate a capability, even if later superseded |
| **Methodological Innovation** | Introduced a novel experimental or theoretical approach |
| **Open Ecosystem Impact** | Enabled broader research community participation (e.g., LLaMA) |

## Exclusion Criteria

These argue against inclusion:

| Criterion | Definition |
|-----------|------------|
| **Limited Adoption** | Technique not widely used in practice despite publication |
| **Superseded Quickly** | Replaced by a better approach within 1-2 years |
| **Single-Model Specific** | Only applicable to one company's proprietary models |
| **Incremental Improvement** | Modest performance gains without conceptual novelty |
| **System Card Only** | Reports capabilities without methodological contribution |

## Category Definitions

### Main Body (Parts I-VII)

Papers introducing foundational techniques with lasting influence. These form the core intellectual lineage of LLMs.

- **Part I:** Neural Beginnings & Learning Mechanisms (1943-1990)
- **Part II:** Sequence Models & Word Embeddings (1990-2013)
- **Part III:** Deep Learning & Attention (2012-2015)
- **Part IV:** The Transformer Era and Pretraining Revolution (2016-2019)
- **Part V:** Emergence and Scale (2019-2020)
- **Part VI:** Efficiency, Alignment, and Reasoning (2021-2022)
- **Part VII:** Open LLMs and Modern Frontier (2023-2024)

### Appendix A: Emerging Results

Recent papers on safety, interpretability, efficiency, and alternative architectures with promising but unproven long-term impact. Papers move here when:
- Too recent to assess adoption
- Address important emerging concerns (safety, interpretability)
- Propose alternatives to dominant paradigms (e.g., SSMs vs Transformers)

### Appendix B: Foundations of Agents

Papers on tool use, planning, reasoning, and agent architectures. Separated because agentic AI represents a distinct application paradigm built on top of LLM foundations.

### Appendix C: System Reports

Technical reports and system cards documenting how foundational techniques are integrated into production systems. These are explicitly *not* foundational research but serve as valuable case studies showing technique integration at scale.

**Inclusion in Appendix C requires:**
- Major industry milestone (GPT-4, Claude 3, Gemini)
- Demonstrates novel integration of multiple techniques
- Publicly available technical documentation

## Decision Flowchart

```
1. Is it a research paper with methodology and experiments?
   NO  -> Does not qualify for main body
         (Consider Appendix C for system reports, or reference in existing paper summary)
   YES -> Continue

2. Does it meet at least 2 PRIMARY criteria?
   NO  -> Does not qualify for main body
         (Consider Appendix A if emerging/promising)
   YES -> Continue

3. Are there EXCLUSION criteria that apply?
   YES -> Weigh against primary criteria
         (Strong exclusion may override 2 primary criteria)
   NO  -> Include in appropriate Part

4. Determine placement:
   - Historical foundation (pre-2017) -> Parts I-III
   - Transformer/pretraining era -> Parts IV-V
   - Efficiency/alignment/scale -> Parts VI-VII
   - Too recent to assess -> Appendix A
   - Agent-specific -> Appendix B
```

## Handling Borderline Cases

Papers marked "(marginal)" in the analysis database meet the minimum criteria but have notable counter-arguments. When evaluating borderline papers:

1. **Prefer inclusion** if the paper fills a gap in the historical narrative
2. **Prefer exclusion** if a more influential paper covers the same ground
3. **Consider moving to appendix** if impact is unclear due to recency

## Versioning

This document should be updated when:
- A new paper is added or removed
- Criteria are refined based on new understanding
- Category definitions evolve

Last updated: January 2026
