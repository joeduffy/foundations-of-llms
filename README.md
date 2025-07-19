# The Foundations of Large Language Models, 1943-2025

A comprehensive collection of the 55 most foundational papers in the development of large language models, spanning from McCulloch-Pitts neurons (1943) to modern reasoning systems (2025).

## Overview

This book assembles the essential papers that shaped the development of large language models. Each paper represents a genuine breakthrough moment—introducing transformative concepts rather than incremental improvements. The collection is organized chronologically to trace the intellectual progression from basic neural networks through modern transformer architectures.

## Structure

### Core Foundations (42 papers)
- **Part I**: Neural Beginnings & Learning Mechanisms (1943–1990) - 4 papers
- **Part II**: Sequence Models & Word Embeddings (1990–2013) - 4 papers  
- **Part III**: Attention and Sequence-to-Sequence Modeling (2014–2016) - 8 papers
- **Part IV**: The Transformer Era and Pretraining Revolution (2017–2019) - 5 papers
- **Part V**: Emergence and Scale (2019–2020) - 5 papers
- **Part VI**: Efficiency, Alignment, and Reasoning (2021–2022) - 9 papers
- **Part VII**: Open LLMs and Modern Frontier (2023–2024) - 7 papers

### Appendices (13 papers)
- **Appendix A**: Emerging Results (2023–2025) - 7 papers (safety, efficiency, reasoning)
- **Appendix B**: Foundations of Agents (2022–2025) - 6 papers

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
├── content/               # LaTeX source files for each section
├── pdfs/                  # Downloaded paper PDFs
├── out/                   # Generated output files
├── Makefile              # Build automation
└── README.md             # This file
```

## Paper Collection

The papers were selected using rigorous criteria:
- **Foundational Impact**: Introduced transformative concepts that changed the field
- **Lasting Influence**: Demonstrated sustained impact on subsequent research
- **Public Availability**: Accessible for academic study and reproduction
- **Technical Depth**: Provided substantive technical contributions

See `final_curated_paper_list.md` for the complete list with URLs and justifications.

## Contributing

This is a curated academic collection. The paper selection has been finalized based on historical impact and foundational importance. For suggestions or corrections, please open an issue.

## License

The LaTeX source code and organizational content are provided under MIT License. Individual papers retain their original copyright and are included for academic purposes under fair use.

## Acknowledgments

This collection builds upon decades of research by hundreds of contributors to the field of artificial intelligence. Special recognition goes to the pioneering researchers whose work made modern large language models possible.