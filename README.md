# The Foundations of Large Language Models, 1943-2026

A comprehensive collection of the foundational papers in the development of large language models, spanning from McCulloch-Pitts neurons (1943) to modern reasoning systems (2026).

## Overview

This book assembles the essential papers that shaped the development of large language models. Each paper represents a genuine breakthrough moment—introducing transformative concepts rather than incremental improvements. The collection is organized chronologically to trace the intellectual progression from basic neural networks through modern transformer architectures.

## Structure

### Core Foundations (56 papers)
- **Part I**: Neural Beginnings & Learning Mechanisms (1943–1990) - 5 papers
- **Part II**: Sequence Models & Word Embeddings (1997–2013) - 6 papers
- **Part III**: Deep Learning & Attention (2012–2015) - 10 papers
- **Part IV**: The Transformer Era and Pretraining Revolution (2016–2019) - 9 papers
- **Part V**: Emergence and Scale (2019–2020) - 8 papers
- **Part VI**: Efficiency, Alignment, and Reasoning (2021–2022) - 10 papers
- **Part VII**: Open LLMs and Modern Frontier (2023–2024) - 8 papers

### Appendices (15 papers)
- **Appendix A**: Emerging Results (2023–2025) - 5 papers
- **Appendix B**: Foundations of Agents (2022–2024) - 4 papers
- **Appendix C**: System Reports & Production Breakthroughs (2023–2025) - 6 papers

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
4. For system reports: it documents a qualitatively new capability or paradigm (not just another strong model).

A paper does **not** belong if:

- It is incremental over an already-included paper with no distinct lasting technique.
- It is a survey or review (not a primary contribution).
- It is overly vendor-specific without advancing the broader state of the art.
- It is too recent to have proven its significance and is not already referenced by frontier systems.

## Contributing

This is a curated academic collection. For suggestions or corrections, please open an issue.

## License

The LaTeX source code and organizational content are provided under MIT License. Individual papers retain their original copyright and are included for academic purposes under fair use.

## Acknowledgments

This collection builds upon decades of research by hundreds of contributors to the field of artificial intelligence. Special recognition goes to the pioneering researchers whose work made modern large language models possible.