This repo contains a book called

> The Foundations of Large Language Models, 1943 - 2026

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
contained in the chronology are welcome. We should not skip any monumental papers even if it means the book is long.

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

The LaTeX content in `content/` is the source of truth for which papers are included. The criteria and structure
below guide paper selection decisions.

# Paper Selection Criteria

A paper belongs in the book if it meets **at least one** of these criteria:

1. It pioneered a technique or architecture that is a direct ancestor of modern LLMs.
2. It is a necessary stepping stone — later foundational work could not have happened without it.
3. It introduced a component, method, or insight now standard in frontier models (evidenced by citations in system
   cards for GPT-4, Gemini, Claude, DeepSeek, Llama, etc.).
4. For system reports: it documents a qualitatively new capability or paradigm (not just another strong model).

A paper does **not** belong if:

- It is incremental over an already-included paper with no distinct lasting technique.
- It is a survey or review (not a primary contribution).
- It is overly vendor-specific without advancing the broader state of the art.
- It is too recent to have proven its significance and is not already referenced by frontier systems.

# Book Structure

The book is organized into seven chronological parts and three appendices:

- **Part I** – Neural Beginnings & Learning Mechanisms (1943–1990)
- **Part II** – Sequence Models & Word Embeddings (1997–2013)
- **Part III** – Deep Learning & Attention (2012–2015)
- **Part IV** – The Transformer Era and Pretraining Revolution (2016–2019)
- **Part V** – Emergence and Scale (2019–2020)
- **Part VI** – Efficiency, Alignment, and Reasoning (2021–2022)
- **Part VII** – Open LLMs and Modern Frontier (2023–2024)
- **Appendix A** – Emerging Results (2023–2024)
- **Appendix B** – Foundations of Agents (2022–2024)
- **Appendix C** – System Reports & Production Breakthroughs (2023–2025)

Each part has a `content/partN-intro.tex` (prose), `content/partN-papers.tex` (paper includes), and individual
`content/summary-*.tex` files for per-paper commentary. The `content/partN-papers.tex` files are the authoritative
list of which papers appear in each section.
