# Changelog

## Second Edition — August 1, 2026

Added **Part VIII: Reasoning and the Open Frontier (2024–2026)**, eight papers covering the
foundational methods of the reasoning era: DeepSeekMath (GRPO), multi-token prediction,
auxiliary-loss-free MoE load balancing, Tülu 3 (RLVR), DeepSeek-R1 (promoted from the former
Appendix C), Muon, Kimi Linear (Kimi Delta Attention), and LatentMoE. Added foundational papers to
earlier parts: knowledge distillation (Part III), AdamW (Part IV), RMSNorm (Part V), speculative
decoding (Part VI), and LLM-QAT (Part VII). Extended Appendix B through the agentic era: Reflexion,
SWE-agent, and Search-R1.

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

**Renamed Appendix A from "Emerging Results" to "The Watch List"** and made it an explicitly
provisional, re-tended section rather than a one-time label: at every edition, papers whose
significance has settled graduate into the main chronological sequence, papers still genuinely
undecided stay, and new candidates are added only if they clear the same bar as everything else in
the book (policy codified in CLAUDE.md). This edition, Mamba (2023) graduated into Part VII — its
significance is no longer in question now that hybrid linear-attention designs descended from it
run in frontier production models (Part VIII). One new candidate was added in its place after a
deliberately hard search: Anthropic's circuit-tracing interpretability work was the strongest
thematic fit — the direct successor to the sparse-autoencoder paper already in the appendix — but
exists only as an interactive essay on Anthropic's own site with no citable PDF, and generating a
facsimile from a webpage screenshot would have broken the standard every other paper in this book
meets. Synthetic Bootstrapped Pretraining (Yang et al., 2025, Apple/Stanford) took the slot instead:
a credibly authored response to the data-scarcity constraint every frontier lab now names, whose
staying power is genuinely undecided. Appendix B (agents) was considered for a similar
restructuring but deliberately left alone: it's organized by topic, not by uncertainty, and agent
research is still developing as a field, which argues for letting it keep growing as a distinct
section rather than promoting or dissolving it.

Added three papers addressing published, non-report research from proprietary labs that
plausibly underlies frontier reasoning and agentic behavior, each argued individually against the
book's selection criteria rather than added on the strength of being frontier-lab work generally:
Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021, Part VI) is the origin of
verifier-based reasoning and GSM8K; Let's Verify Step by Step (Lightman et al., 2023, Part VII)
shows that step-level reward models select correct reasoning more reliably than outcome-level
ones; WebGPT (Nakano et al., 2021) is the origin of tool-augmented,
human-feedback-trained agents and is now Appendix B's opening paper, predating and underlying
Search-R1. Two other real candidates (Self-Consistency, Scaling Laws for Reward Model
Overoptimization) were considered and did not clear the bar as standalone embedded papers; the
former is now properly credited in the existing chain-of-thought summary instead.

Added nucleus sampling (Holtzman et al., 2019) to Part V, since top-p is a standard generation
control that the book did not previously cover. Added two Watch List papers: mHC (Xie et al.,
2025), the residual-connection design adopted in DeepSeek-V4, and chain-of-thought monitoring
(Baker et al., 2025), which the GPT-5 system card cites in its commitment to monitorable
reasoning. Updated Part VIII and the Epilogue for DeepSeek-V4's compressed-attention route to
million-token context and for reward hacking as a failure mode of verifiable rewards. Net: 86
papers (71 core, 15 appendix).

## First Edition — February 18, 2026

Initial release: 71 papers across seven chronological parts and three appendices,
spanning 1943–2025.
