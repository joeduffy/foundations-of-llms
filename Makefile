# Makefile for "The Foundations of Large Language Models" book

# Variables
MAIN = main
OUTPUT_DIR = out
CONTENT_DIR = content
PDF_DIR = pdfs
PDFLATEX = /Library/TeX/texbin/pdflatex

# Add TeX binaries to PATH
export PATH := /Library/TeX/texbin:$(PATH)

# Default target
all: book

# Build the complete book
book: setup
	@echo "Building the book..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	@echo "Book built successfully: $(OUTPUT_DIR)/$(MAIN).pdf"

# Create necessary directories and check dependencies
setup:
	@echo "Setting up build environment..."
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(PDF_DIR)
	@mkdir -p $(CONTENT_DIR)
	@echo "Checking for pdflatex..."
	@which pdflatex > /dev/null || (echo "Error: pdflatex not found. Please install LaTeX." && exit 1)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OUTPUT_DIR)/*.aux $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.toc $(OUTPUT_DIR)/*.out $(OUTPUT_DIR)/*.fdb_latexmk $(OUTPUT_DIR)/*.fls

# Clean everything including the final PDF
distclean: clean
	@echo "Removing all generated files..."
	@rm -f $(OUTPUT_DIR)/$(MAIN).pdf

# Download papers (placeholder - URLs would need to be added)
download-papers:
	@echo "Downloading papers..."
	@echo "Note: This requires implementing individual download commands for each paper"
	@echo "Papers should be saved to $(PDF_DIR)/ with appropriate filenames"

# Quick build without double compilation (for development)
quick: setup
	@echo "Quick build (single pass)..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex

# Watch for changes and rebuild (requires entr or inotify-tools)
watch:
	@echo "Watching for changes..."
	@find . -name "*.tex" | entr make quick

# Help
help:
	@echo "Available targets:"
	@echo "  all (default)    - Build the complete book"
	@echo "  book             - Build the complete book"
	@echo "  quick            - Quick build (single pass)"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Remove all generated files"
	@echo "  download-papers  - Download all required papers"
	@echo "  watch            - Watch for changes and rebuild"
	@echo "  help             - Show this help message"

.PHONY: all book setup clean distclean download-papers quick watch help