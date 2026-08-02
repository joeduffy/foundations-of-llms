# Changelog

## Second Edition — August 1, 2026

Added **Part VIII: Reasoning and the Open Frontier (2024–2026)**, eight papers covering the
foundational methods of the reasoning era: DeepSeekMath (GRPO), multi-token prediction,
auxiliary-loss-free MoE load balancing, Tülu 3 (RLVR), DeepSeek-R1 (promoted from the former
Appendix C), Muon, Kimi Linear (Kimi Delta Attention), and LatentMoE. Added foundational papers to
earlier parts: knowledge distillation (Part III), AdamW (Part IV), RMSNorm (Part V), speculative
decoding (Part VI), and LLM-QAT (Part VII). Extended Appendix B through the agentic era: Reflexion,
SWE-agent, and Search-R1. Adopted a stricter selection policy (see README): integrated model
reports and system cards are cited, not embedded. Removed Appendix C (GPT-4 Technical Report,
Gemini 1.5, DeepSeek-V3, o1 System Card, GPT-5 System Card) and, from Part VII, Llama 2 and
Mixtral — their lasting techniques are documented by embedded method papers. Refreshed the
prologue and epilogue for the reasoning era; extended the concept family tree with a Part 8 row;
corrected era ranges. Net: 79 papers (67 core, 12 appendix).

Fixed two structural bugs present since the first edition:

- Every one of the book's per-paper table-of-contents entries and cross-references (all 79) linked
  to the entry immediately preceding it rather than to itself. Root cause: each entry's hyperref
  anchor was created before the page break that started its own page, binding it to whatever
  anchor the previous entry had left current. Fixed by moving anchor creation into the
  `papersummary` environment, immediately after its page break. Verified against the built PDF's
  actual link-annotation destinations, not just page numbers.
- `\chapter*` never calls `\chaptermark`, so the running head froze after any unnumbered chapter
  and could show a stale or flatly wrong section name (e.g. "Contents" bleeding into the Prologue)
  for hundreds of pages. Fixed with two macros, `\PlainChapter` and `\PartIntroChapter`, that set
  the running-head marks explicitly and correctly for standalone chapters versus chapters nested
  inside a part.

Also fixed the LLaMA (2023) facsimile, which had inadvertently duplicated the Llama 2 PDF.

A typography and layout pass toward the production standard of a bound mathematics monograph:

- **Twoside layout.** Mirrored inner/outer margins for binding; canonical running head with the
  page number on the outer edge, the enclosing part's title on the verso, and the current paper's
  author/year on the recto.
- **Fixed double pagination on every embedded paper.** Facsimile pages previously carried both our
  own header/page-number and the original publication's own masthead and pagination, competing for
  attention. Facsimile pages now suppress our header entirely.
- Switched to single spacing; muted the saturated hyperlink blue to a near-black ink color
  throughout; removed the version stamp and email address from the title page in favor of a proper
  copyright/colophon page.
- Fixed the table of contents so a part's introduction nests visibly beneath its part and above
  its papers, rather than sitting flush with one and unindented relative to the other; shortened
  part titles used in the TOC and running heads so they no longer wrap awkwardly.
- Removed the forced blank page that had appeared between every part's title page and its own
  introduction (`openright` forced both onto separate odd pages; reverted to `openany`, keeping
  `twoside` for the margin and header benefits).
- **Added a back-of-book index**: every paper's authors, plus roughly fifty key technical terms
  indexed at their defining introduction (attention, transformer, backpropagation, RLHF, GRPO,
  quantization-aware training, and so on).

**Rewrote the prologue and epilogue as a matched pair.** They had been revised independently on
different schedules and it showed: the prologue's narrative stopped at pretraining-era scale and
never mentioned alignment, reasoning, or agency even though Part VIII and half of Appendix B are
now devoted to exactly that; the epilogue never mentioned agents at all despite Appendix B's
existence, and posed a different central question than the one the prologue opens with, never
reconciling the two. Both now share one throughline. Added a new prologue section, "Alignment,
Reasoning, and Agency," carrying the book's argument forward through RLHF, DPO, GRPO/verifiable
rewards, and trained agentic RL. The epilogue now opens by explicitly returning to the prologue's
question, adds a new "Agency and Reliability" section on the reliability problems specific to
long-horizon agents, and closes on a specific, falsifiable open question about reasoning-chain
faithfulness rather than a general historical observation. Also de-formularized the epilogue's
section endings, which had mechanically repeated the same "list, then negate" closing shape across
most of its five sections.

## First Edition — February 18, 2026

Initial release: 71 papers across seven chronological parts and three appendices,
spanning 1943–2025.
