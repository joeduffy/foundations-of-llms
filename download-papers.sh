#!/bin/bash

# Download script for "The Foundations of Large Language Models" papers
# This script downloads all 55 papers to the pdfs/ directory with consistent naming

set -e  # Exit on any error

echo "Starting download of 55 foundational papers..."

# Create pdfs directory if it doesn't exist
mkdir -p pdfs

# Function to download with proper naming
download_paper() {
    local url="$1"
    local filename="$2"
    local title="$3"
    
    echo "Downloading: $title"
    echo "  URL: $url"
    echo "  File: $filename"
    
    if [ ! -f "pdfs/$filename" ]; then
        curl -L -o "pdfs/$filename" "$url" || {
            echo "  ❌ Failed to download $filename"
            return 1
        }
        echo "  ✅ Downloaded successfully"
    else
        echo "  ⏭️  Already exists, skipping"
    fi
    echo ""
}

# Part I: Neural Beginnings & Learning Mechanisms (1943–1990)
download_paper "https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf" \
    "mcculloch-pitts-1943.pdf" \
    "McCulloch & Pitts (1943) - A Logical Calculus"

download_paper "https://www.cse.chalmers.se/~coquand/AIP/perceptron.pdf" \
    "rosenblatt-1958.pdf" \
    "Rosenblatt (1958) - The Perceptron"

download_paper "https://www.pnas.org/doi/pdf/10.1073/pnas.79.8.2554" \
    "hopfield-1982.pdf" \
    "Hopfield (1982) - Neural Networks and Physical Systems"

download_paper "https://www.cs.toronto.edu/~fritz/absps/naturebp.pdf" \
    "rumelhart-hinton-williams-1986.pdf" \
    "Rumelhart, Hinton & Williams (1986) - Backpropagation"

# Part II: Sequence Models & Word Embeddings (1990–2013)
download_paper "https://www.bioinf.jku.at/publications/older/2604.pdf" \
    "hochreiter-schmidhuber-1997.pdf" \
    "Hochreiter & Schmidhuber (1997) - LSTM"

download_paper "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf" \
    "bengio-2003.pdf" \
    "Bengio et al. (2003) - Neural Probabilistic Language Model"

download_paper "https://arxiv.org/pdf/1301.3781.pdf" \
    "mikolov-2013.pdf" \
    "Mikolov et al. (2013) - Word2Vec"

download_paper "https://arxiv.org/pdf/1308.0850.pdf" \
    "graves-2013.pdf" \
    "Graves (2013) - Generating Sequences with RNNs"

# Part III: Attention and Sequence-to-Sequence Modeling (2014–2016)
download_paper "https://arxiv.org/pdf/1412.6980.pdf" \
    "kingma-ba-2014.pdf" \
    "Kingma & Ba (2014) - Adam Optimizer"

download_paper "https://papers.nips.cc/paper_files/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf" \
    "sutskever-2014.pdf" \
    "Sutskever et al. (2014) - Sequence to Sequence Learning"

download_paper "https://arxiv.org/pdf/1409.0473.pdf" \
    "bahdanau-2014.pdf" \
    "Bahdanau et al. (2014) - Neural Machine Translation with Attention"

download_paper "https://nlp.stanford.edu/pubs/glove.pdf" \
    "pennington-2014.pdf" \
    "Pennington et al. (2014) - GloVe"

download_paper "https://arxiv.org/pdf/1406.1078.pdf" \
    "cho-2014.pdf" \
    "Cho et al. (2014) - RNN Encoder-Decoder"

download_paper "https://arxiv.org/pdf/1512.03385.pdf" \
    "he-2015.pdf" \
    "He et al. (2015) - ResNet"

download_paper "https://arxiv.org/pdf/1508.04025.pdf" \
    "luong-2015.pdf" \
    "Luong et al. (2015) - Effective Approaches to Attention"

download_paper "https://arxiv.org/pdf/1508.07909.pdf" \
    "sennrich-2015.pdf" \
    "Sennrich et al. (2015) - Neural Machine Translation of Rare Words"

# Part IV: The Transformer Era and Pretraining Revolution (2017–2019)
download_paper "https://arxiv.org/pdf/1706.03762.pdf" \
    "vaswani-2017.pdf" \
    "Vaswani et al. (2017) - Attention Is All You Need"

download_paper "https://arxiv.org/pdf/1802.05365.pdf" \
    "peters-2018.pdf" \
    "Peters et al. (2018) - ELMo"

download_paper "https://arxiv.org/pdf/1810.04805.pdf" \
    "devlin-2018.pdf" \
    "Devlin et al. (2018) - BERT"

download_paper "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" \
    "radford-2018.pdf" \
    "Radford et al. (2018) - GPT-1"

download_paper "https://arxiv.org/pdf/1910.10683.pdf" \
    "raffel-2019.pdf" \
    "Raffel et al. (2019) - T5"

# Part V: Emergence and Scale (2019–2020)
download_paper "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" \
    "radford-2019.pdf" \
    "Radford et al. (2019) - GPT-2"

download_paper "https://arxiv.org/pdf/1907.11692.pdf" \
    "liu-2019.pdf" \
    "Liu et al. (2019) - RoBERTa"

download_paper "https://arxiv.org/pdf/2005.14165.pdf" \
    "brown-2020.pdf" \
    "Brown et al. (2020) - GPT-3"

download_paper "https://arxiv.org/pdf/2001.08361.pdf" \
    "kaplan-2020.pdf" \
    "Kaplan et al. (2020) - Scaling Laws"

download_paper "https://arxiv.org/pdf/2003.10555.pdf" \
    "clark-2020.pdf" \
    "Clark et al. (2020) - ELECTRA"

# Part VI: Efficiency, Alignment, and Reasoning (2021–2022)
download_paper "https://arxiv.org/pdf/2002.05202.pdf" \
    "shazeer-2020.pdf" \
    "Shazeer (2020) - GLU Variants"

download_paper "https://arxiv.org/pdf/2106.09685.pdf" \
    "hu-2021.pdf" \
    "Hu et al. (2021) - LoRA"

download_paper "https://arxiv.org/pdf/2205.14135.pdf" \
    "dao-2022.pdf" \
    "Dao et al. (2022) - FlashAttention"

download_paper "https://arxiv.org/pdf/2203.02155.pdf" \
    "ouyang-2022.pdf" \
    "Ouyang et al. (2022) - InstructGPT"

download_paper "https://arxiv.org/pdf/2201.11903.pdf" \
    "wei-2022.pdf" \
    "Wei et al. (2022) - Chain-of-Thought"

download_paper "https://arxiv.org/pdf/2212.08073.pdf" \
    "bai-2022.pdf" \
    "Bai et al. (2022) - Constitutional AI"

download_paper "https://arxiv.org/pdf/2210.03629.pdf" \
    "yao-2022.pdf" \
    "Yao et al. (2022) - ReAct"

download_paper "https://arxiv.org/pdf/2203.15556.pdf" \
    "hoffmann-2022.pdf" \
    "Hoffmann et al. (2022) - Chinchilla"

download_paper "https://arxiv.org/pdf/2204.02311.pdf" \
    "chowdhery-2022.pdf" \
    "Chowdhery et al. (2022) - PaLM"

# Part VII: Open LLMs and Modern Frontier (2023–2024)
download_paper "https://arxiv.org/pdf/2303.08774.pdf" \
    "openai-2023.pdf" \
    "OpenAI (2023) - GPT-4 Technical Report"

download_paper "https://arxiv.org/pdf/2305.14314.pdf" \
    "dettmers-2023.pdf" \
    "Dettmers et al. (2023) - QLoRA"

download_paper "https://arxiv.org/pdf/2307.09288.pdf" \
    "touvron-2023.pdf" \
    "Touvron et al. (2023) - Llama 2"

download_paper "https://arxiv.org/pdf/2308.12950.pdf" \
    "roziere-2023.pdf" \
    "Rozière et al. (2023) - Code Llama"

download_paper "https://arxiv.org/pdf/2401.04088.pdf" \
    "mistral-2024.pdf" \
    "Mistral AI (2024) - Mixtral of Experts"

download_paper "https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf" \
    "anthropic-2024.pdf" \
    "Anthropic (2024) - Claude 3 Model Family"

download_paper "https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf" \
    "deepmind-2024.pdf" \
    "DeepMind (2024) - Gemini 1.5"

# Appendix A: Emerging Results (2023–2025)
download_paper "https://arxiv.org/pdf/2310.12773.pdf" \
    "bai-2023.pdf" \
    "Bai et al. (2023) - Safe RLHF"

download_paper "https://arxiv.org/pdf/2309.08600.pdf" \
    "cunningham-2023.pdf" \
    "Cunningham et al. (2023) - Sparse Autoencoders"

download_paper "https://arxiv.org/pdf/2403.04893.pdf" \
    "casper-2024.pdf" \
    "Casper et al. (2024) - Safe Harbor for AI Evaluation"

download_paper "https://arxiv.org/pdf/2412.16720.pdf" \
    "openai-2024-o1.pdf" \
    "OpenAI (2024) - o1 System Card"

download_paper "https://arxiv.org/pdf/2312.00752.pdf" \
    "gu-2023.pdf" \
    "Gu & Dao (2023) - Mamba"

download_paper "https://arxiv.org/pdf/2402.07871.pdf" \
    "frantar-2024.pdf" \
    "Frantar et al. (2024) - Scaling Laws for Fine-Grained MoE"

download_paper "https://arxiv.org/pdf/2503.06692.pdf" \
    "yan-2025.pdf" \
    "Yan et al. (2025) - InftyThink"

# Appendix B: Foundations of Agents (2022–2025)
download_paper "https://arxiv.org/pdf/2209.07753.pdf" \
    "liang-2022.pdf" \
    "Liang et al. (2022) - Code as Policies"

download_paper "https://arxiv.org/pdf/2302.04761.pdf" \
    "schick-2023.pdf" \
    "Schick et al. (2023) - Toolformer"

download_paper "https://arxiv.org/pdf/2305.10601.pdf" \
    "yao-2023.pdf" \
    "Yao et al. (2023) - Tree of Thoughts"

download_paper "https://arxiv.org/pdf/2402.01030.pdf" \
    "wang-2024.pdf" \
    "Wang et al. (2024) - Executable Code Actions"

download_paper "https://arxiv.org/pdf/2503.09572.pdf" \
    "erdogan-2025.pdf" \
    "Erdogan et al. (2025) - Plan-and-Act"

download_paper "https://arxiv.org/pdf/2504.01990.pdf" \
    "liu-2025.pdf" \
    "Liu et al. (2025) - Advances and Challenges in Foundation Agents"

echo "=========================================="
echo "Download complete!"
echo "Downloaded papers to pdfs/ directory"
echo "Total papers: 55"
echo ""
echo "Next steps:"
echo "1. Verify all downloads completed successfully"
echo "2. Run 'make book' to build the complete book"
echo "=========================================="