# The Foundations of Large Language Models, 1943-2026

⬇️ **[PDF (Archive.org)](https://archive.org/download/foundations-of-llms/FoundationsOfLLMs.pdf)** | ⬇️ **[PDF (Google Drive)](https://drive.google.com/file/d/1BEbhrt5D63V2clS3HYnZWUhrGykvDtmk/view?usp=drive_link)**

A comprehensive collection of the foundational papers in the development of large language models, spanning from McCulloch-Pitts neurons (1943) to modern reasoning systems (2026).

## Overview

This book assembles the essential papers that shaped the development of large language models. Each paper represents a genuine breakthrough moment—introducing transformative concepts rather than incremental improvements. The collection is organized chronologically to trace the intellectual progression from basic neural networks through modern transformer architectures.

## Structure

### Core Foundations (67 papers)
- **Part I**: Neural Beginnings & Learning Mechanisms (1943–1990) - 5 papers
- **Part II**: Sequence Models & Word Embeddings (1997–2013) - 6 papers
- **Part III**: Deep Learning & Attention (2012–2015) - 11 papers
- **Part IV**: The Transformer Era and Pretraining Revolution (2016–2019) - 10 papers
- **Part V**: Emergence and Scale (2019–2020) - 9 papers
- **Part VI**: Efficiency, Alignment, and Reasoning (2021–2022) - 11 papers
- **Part VII**: Open LLMs and Modern Frontier (2023–2024) - 7 papers
- **Part VIII**: Reasoning and the Open Frontier (2024–2026) - 8 papers

### Appendices (12 papers)
- **Appendix A**: Emerging Results (2023–2025) - 5 papers
- **Appendix B**: Foundations of Agents (2022–2025) - 7 papers

## Building the Book

### Prerequisites
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Make (for build automation)

### Quick Start
```bash
# Build the complete book
make book

# Quick development build
make quick

# Clean build artifacts
make clean
```

### Directory Structure
```
├── main.tex              # Main LaTeX document
├── content/              # LaTeX source files for each section
├── pdfs/                 # Paper PDFs
├── out/                  # Generated output files
├── Makefile              # Build automation
└── README.md             # This file
```

## Paper Selection Criteria

A paper belongs in the book if it meets at least one of the following:

1. It pioneered a technique or architecture that is a direct ancestor of modern LLMs.
2. It is a necessary stepping stone — later foundational work could not have happened without it.
3. It introduced a component, method, or insight now standard in frontier models (evidenced by citations in system cards for GPT-4, Gemini, Claude, DeepSeek, Llama, etc.).

A paper does **not** belong if:

- It is a model report or system card whose primary content is capabilities and evaluations. The book embeds the foundational technique paper behind a model's innovations and cites the model report in prose (a model paper qualifies only if it *introduced* a foundational technique or paradigm that has no standalone source).
- It is incremental over an already-included paper with no distinct lasting technique.
- It is a survey or review (not a primary contribution).
- It is overly vendor-specific without advancing the broader state of the art.
- It is too recent to have proven its significance and is not already referenced by frontier systems.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the revision history.

## Contributing

This is a curated academic collection. For suggestions or corrections, please open an issue.

## License

The LaTeX source and the original prose in this collection are provided under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (see [LICENSE](LICENSE)). Individual papers retain their original copyright and are included for academic purposes under fair use.

## Acknowledgments

This collection builds upon decades of research by hundreds of contributors to the field of artificial intelligence. Special recognition goes to the pioneering researchers whose work made modern large language models possible.