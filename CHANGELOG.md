# Changelog

## Second Edition — August 1, 2026

Added **Part VIII: Reasoning and the Open Frontier (2024–2026)**, eight papers covering the
foundational methods of the reasoning era: DeepSeekMath (GRPO), multi-token prediction,
auxiliary-loss-free MoE load balancing, Tülu 3 (RLVR), DeepSeek-R1 (promoted from the former
Appendix C), Muon, Kimi Linear (Kimi Delta Attention), and LatentMoE. Added foundational papers to
earlier parts: knowledge distillation (Part III), AdamW (Part IV), RMSNorm (Part V), speculative
decoding (Part VI), and LLM-QAT (Part VII). Extended Appendix B through the agentic era: Reflexion,
SWE-agent, and Search-R1. Net: 79 papers (67 core, 12 appendix).

Adopted a stricter selection policy (see README): the book now embeds foundational technique
papers only, never model reports or system cards. Removed Appendix C (GPT-4 Technical Report,
Gemini 1.5, DeepSeek-V3, o1 System Card, GPT-5 System Card) and, from Part VII, Llama 2 and
Mixtral — their lasting techniques are documented by embedded method papers instead. Also fixed
the LLaMA (2023) facsimile, which had inadvertently duplicated the Llama 2 PDF.

**Rewrote the prologue and epilogue as a matched pair.** The prologue's argument now carries
forward through RLHF, DPO, GRPO/verifiable rewards, and trained agentic RL rather than stopping at
pretraining-era scale, via a new "Alignment, Reasoning, and Agency" section. The epilogue opens by
returning explicitly to the prologue's central question, adds a new "Agency and Reliability"
section on the reliability problems specific to long-horizon agents, and closes on a specific,
falsifiable open question about reasoning-chain faithfulness rather than a general historical
observation.

Also reset the book's typography and layout toward the production standard of a bound mathematics
monograph — twoside layout with canonical running heads, single spacing, a muted ink color for
hyperlinks, a proper copyright/colophon page, and a back-of-book index — and fixed a number of
structural defects along the way: duplicated table-of-contents anchors and cross-references
(every per-paper entry linked to the one before it, not itself), a running head that froze after
unnumbered chapters, inconsistent table-of-contents nesting, several stray forced blank pages, and
a title page that briefly overflowed onto a second sheet after a tikz bounding-box regression.

## First Edition — February 18, 2026

Initial release: 71 papers across seven chronological parts and three appendices,
spanning 1943–2025.
