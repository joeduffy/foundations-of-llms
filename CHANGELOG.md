# Changelog

## v2.0.1 — August 1, 2026

- Fixed a structural bug, present since v1.0.0, in which every one of the book's per-paper table-
  of-contents entries and cross-references (all 79) linked to the entry immediately preceding it
  rather than to itself. Root cause: each entry's hyperref anchor was created before the page break
  that started its own page, binding it to whatever anchor the previous entry had left current.
  Fixed by moving anchor creation into the `papersummary` environment, immediately after its page
  break. Verified against the built PDF's actual link-annotation destinations, not just page
  numbers: 0 of 79 entries mismatched, down from 79 of 79.

## v2.0.0 — August 1, 2026

- Added **Part VIII: Reasoning and the Open Frontier (2024–2026)**, eight papers covering the
  foundational methods of the reasoning era: DeepSeekMath (GRPO), multi-token prediction,
  auxiliary-loss-free MoE load balancing, Tülu 3 (RLVR), DeepSeek-R1 (promoted from the former
  Appendix C), Muon, Kimi Linear (Kimi Delta Attention), and LatentMoE.
- Added foundational papers to earlier parts: knowledge distillation (Part III), AdamW (Part IV),
  RMSNorm (Part V), speculative decoding (Part VI), and LLM-QAT (Part VII).
- Extended Appendix B through the agentic era: Reflexion (verbal self-reflection), SWE-agent
  (agent-computer interfaces), and Search-R1 (reinforcement learning for multi-turn tool use).
- Adopted a stricter selection policy (see README): integrated model reports and system cards are
  cited, not embedded. Removed Appendix C (GPT-4 Technical Report, Gemini 1.5, DeepSeek-V3,
  o1 System Card, GPT-5 System Card) and, from Part VII, Llama 2 and Mixtral — their lasting
  techniques are documented by embedded method papers.
- Fixed the LLaMA (2023) facsimile, which had inadvertently duplicated the Llama 2 PDF.
- Refreshed the prologue and epilogue for the reasoning era; extended the concept family tree with
  a Part 8 row; corrected era ranges (Part II 1997–2013, Part IV 2016–2019, Appendix A 2023–2025,
  Appendix B 2022–2025); added this revision history to the front matter.
- Net: 79 papers (67 core, 12 appendix), roughly 190 fewer pages than v1.0.0.

## v1.0.0 — February 18, 2026

- Initial release: 71 papers across seven chronological parts and three appendices,
  spanning 1943–2025.
