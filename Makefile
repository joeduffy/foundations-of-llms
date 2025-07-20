# Makefile for "The Foundations of Large Language Models" book

# Variables
MAIN = main
MAIN_MOBILE = main-mobile
MAIN_IPAD = main-ipad
OUTPUT_DIR = out
CONTENT_DIR = content
PDF_DIR = pdfs
PDFLATEX = /Library/TeX/texbin/pdflatex

# Add TeX binaries to PATH
export PATH := /Library/TeX/texbin:$(PATH)

# Default target
all: book mobile ipad

# Build the complete book
book: setup
	@echo "Building the book..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	@echo "Book built successfully: $(OUTPUT_DIR)/$(MAIN).pdf"

# Build mobile-optimized version
mobile: setup
	@echo "Building mobile version..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_MOBILE).tex
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_MOBILE).tex
	@echo "Mobile version built successfully: $(OUTPUT_DIR)/$(MAIN_MOBILE).pdf"

# Build iPad-optimized version
ipad: setup
	@echo "Building iPad version..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_IPAD).tex
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_IPAD).tex
	@echo "iPad version built successfully: $(OUTPUT_DIR)/$(MAIN_IPAD).pdf"

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

# Clean everything including the final PDFs
distclean: clean
	@echo "Removing all generated files..."
	@rm -f $(OUTPUT_DIR)/$(MAIN).pdf $(OUTPUT_DIR)/$(MAIN_MOBILE).pdf $(OUTPUT_DIR)/$(MAIN_IPAD).pdf

# Download papers (placeholder - URLs would need to be added)
download-papers:
	@echo "Downloading papers..."
	@echo "Note: This requires implementing individual download commands for each paper"
	@echo "Papers should be saved to $(PDF_DIR)/ with appropriate filenames"

# Quick build without double compilation (for development)
quick: setup
	@echo "Quick build (single pass)..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex

# Quick build for mobile version
quick-mobile: setup
	@echo "Quick build mobile (single pass)..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_MOBILE).tex

# Quick build for iPad version
quick-ipad: setup
	@echo "Quick build iPad (single pass)..."
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN_IPAD).tex

# Watch for changes and rebuild (requires entr or inotify-tools)
watch:
	@echo "Watching for changes..."
	@find . -name "*.tex" | entr make quick

# Help
help:
	@echo "Available targets:"
	@echo "  all (default)    - Build all book versions (main, mobile, iPad)"
	@echo "  book             - Build the complete book"
	@echo "  mobile           - Build mobile-optimized version"
	@echo "  ipad             - Build iPad-optimized version"
	@echo "  quick            - Quick build (single pass) for main book"
	@echo "  quick-mobile     - Quick build for mobile version"
	@echo "  quick-ipad       - Quick build for iPad version"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Remove all generated files including PDFs"
	@echo "  download-papers  - Download all required papers"
	@echo "  watch            - Watch for changes and rebuild"
	@echo "  help             - Show this help message"

.PHONY: all book mobile ipad setup clean distclean download-papers quick quick-mobile quick-ipad watch help