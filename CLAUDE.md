This repo contains a book called

> The Foundations of Large Language Models, 1943 - 2025

It is a collection of the most influential papers in the technology lineage of neural networks leading up to, and
following, the invention of large language models (LLMs) like OpenAI's GPT, Anthropic's Claude, Llama, and more.

The intent is to capture timeless classics and those that have proven to be influential in the development and evolution
of LLMs. This curriculum is meant for highly technical readers already relatively well-versed in AI, machine learning,
and basic neural network techniques, but aims to fill any gaps the reader has in the entire history leading up to the
the now-leading frontier models.

The book is chronologically broken into chapters representing the key eras in the development of LLMs. Each chapter
contains the most influential, impactful, and iconic publicly available papers in the form of PDFs, which advanced the
state of the art during that period.

We are focused largely on contributions in the public domain and less on overly commercial or self-serving publications
that apply only to single companies or single models, although case studies of assembling the rich array of techniques
contained in the chronology are welcome. We are aiming for somewhere around 40 papers, however, it is most important
to be complete. We should not skip any monumental papers even if it presses up against our maximum length.

The book is split into sections:

* *Title page*: A title page with the title of the book, rendered stylistically.

* *Table of Contents*: A simple table of contents with book page numbers.

* *Prologue*: an introductory section explaining the goals of the book, a brief summary of the arc of history, and the
    usual content that a prologue for such a book would contain.

* *The sections themselves*: Each section should have an introductory 1-page describing the era, context, and key
    advances made within it. This introduction is just a few paragraphs of prose, but also lists the set of papers
    within along with a 1-sentence summary of each describing that paper's key ideas and contributions to the field.

* *Epilogue*: an ending section that adds some color to the current and future state of the art, including some
    discussion of the ongoing challenges in the field, such as safety and human alignment, for instance.

All prose in the book is highly technical, focused, and does not use flowery language. It should read like a Springer
Verlag textbook.

The content is stored in content/ and final book will be rendered into out/. Any prose written should be stored in
an appropriate format for rendering into a book, such as LaTeX. The paper PDFs should be downloaded from the Internet,
stored in pdfs/, and embedded directly into the final PDF, interspersed with the prose written with LaTeX rendered
appropriately. It will be up to you to orchestrate whatever tools needed to produce the final PDF.

We have done some preliminary research, so you will find below a starting point for the book. We will work together to
shape it into the final product, which will be a single PDF that we can then print, distribute, and read.

# Preliminary Research

---

## **Part I – Neural Beginnings & Learning Mechanisms (1943–1990)**

1. McCulloch & Pitts (1943) — [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf)
2. Rosenblatt (1958) — [The Perceptron](https://www.cse.chalmers.se/~coquand/AIP/perceptron.pdf)
3. Minsky & Papert (1969) — [Perceptrons](https://web.media.mit.edu/~minsky/papers/Perceptrons1969.pdf)
4. Hopfield (1982) — [Neural Networks and Physical Systems with Emergent Collective Computational Abilities](https://www.pnas.org/doi/pdf/10.1073/pnas.79.8.2554)
5. Rumelhart, Hinton & Williams (1986) — [Learning Representations by Back-Propagating Errors](https://www.cs.toronto.edu/~fritz/absps/naturebp.pdf)

---

## **Part II – Sequence Models & Word Embeddings (1990–2013)**

6. Elman (1990) — [Finding Structure in Time](https://crl.ucsd.edu/~elman/Papers/fsit.pdf)
7. Hochreiter & Schmidhuber (1997) — [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
8. Bengio et al. (2003) — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
9. Mikolov et al. (2010) — [Recurrent Neural Network Based Language Model](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_1045.pdf)
10. Mikolov et al. (2013) — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
11. Pennington et al. (2014) — [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

---

## **Part III – Attention and Sequence-to-Sequence Modeling (2014–2016)**

12. Sutskever et al. (2014) — [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper_files/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
13. Bahdanau et al. (2014) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
14. Sennrich et al. (2015) — [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)

---

## **Part IV – The Transformer Era and Pretraining Revolution (2017–2019)**

15. Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
16. Peters et al. (2018) — [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/pdf/1802.05365.pdf)
17. Howard & Ruder (2018) — [Universal Language Model Fine-Tuning (ULMFiT)](https://arxiv.org/pdf/1801.06146.pdf)
18. Devlin et al. (2018) — [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805.pdf)
19. Radford et al. (2018) — [Improving Language Understanding by Generative Pretraining (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---

## **Part V – Emergence and Scale (2019–2020)**

20. Radford et al. (2019) — [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
21. Brown et al. (2020) — [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/pdf/2005.14165.pdf)
22. Kaplan et al. (2020) — [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
23. Choromanski et al. (2020) — [Rethinking Attention with Performers](https://arxiv.org/pdf/2009.14794.pdf)
24. Lewis et al. (2020) — [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/pdf/2005.11401.pdf)

---

## **Part VI – Efficiency, Alignment, and Reasoning (2021–2022)**

25. Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
26. Su et al. (2021) — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf)
27. Press et al. (2021) — [ALiBi: Position Biases for Longer Sequences](https://arxiv.org/pdf/2108.12409.pdf)
28. Borgeaud et al. (2021) — [Improving Language Models by Retrieving from Trillions of Tokens (RETRO)](https://arxiv.org/pdf/2112.04426.pdf)
29. Fedus et al. (2022) — [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)
30. Dao et al. (2022) — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)
31. Ouyang et al. (2022) — [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/pdf/2203.02155.pdf)
32. Wei et al. (2022) — [Chain-of-Thought Prompting Elicits Reasoning in Language Models](https://arxiv.org/pdf/2201.11903.pdf)
33. Bai et al. (2022) — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)
34. Yao et al. (2022) — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629.pdf)
35. Hoffmann et al. (2022) — [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/pdf/2203.15556.pdf)

---

## **Part VII – Open LLMs and Modern Frontier (2023–2024)**

36. Touvron et al. (2023) — [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)
37. Dettmers et al. (2023) — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)
38. OpenAI (2023) — [GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf)
39. Mistral AI (2024) — [Mixtral of Experts: Sparse Mixture of Experts Model](https://arxiv.org/pdf/2401.04088.pdf)
40. DeepMind (2024) — [Gemini 1.5 Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)

---

## **Appendix A – Emerging Results (2025)**

41. Wang et al. (2025) — [Long-Input Fine-Tuning (LIFT)](https://arxiv.org/pdf/2402.05352.pdf)
42. Zhao et al. (2025) — [RoPE Extrapolation Without Retraining](https://arxiv.org/pdf/2404.06691.pdf)
43. Jiang et al. (2025) — [InftyThink: Curriculum-Based Long-Form Reasoning](https://arxiv.org/pdf/2403.07513.pdf)
44. Li et al. (2025) — [Sparse MoE as Unified Competitive Learning (SMoE)](https://arxiv.org/pdf/2403.09674.pdf)
45. Zhang et al. (2025) — [Mixture of Grouped Experts (MoGE)](https://arxiv.org/pdf/2405.04122.pdf)
46. Chen et al. (2025) — [Private Steering for Alignment (PSA)](https://arxiv.org/pdf/2401.06086.pdf)
47. Li et al. (2025) — [Reasoning-as-Logic-Units (RLU)](https://arxiv.org/pdf/2402.02862.pdf)

---
