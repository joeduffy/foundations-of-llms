# Paper Analysis Database

This document records the analysis and decision for every paper considered for inclusion in "The Foundations of Large Language Models, 1943-2025."

**Status codes:**
- `INCLUDED` - Currently in the book
- `EXCLUDED` - Considered and rejected
- `CANDIDATE` - Under consideration for addition

**Verdict codes:**
- `KEEP` - Retain in current location
- `REMOVE` - Remove from book
- `ADD` - Add to book
- `MOVE` - Relocate to different section

---

## Part I: Neural Beginnings & Learning Mechanisms (1943-1990)

### Included Papers

#### McCulloch & Pitts (1943)
**Title:** A Logical Calculus of the Ideas Immanent in Nervous Activity
**Status:** INCLUDED
**PDF:** `mcculloch-pitts-1943.pdf`
**Criteria Met:** Foundational Impact, Historical Significance
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Established that networks of simple computational units can perform arbitrary logical operations. Foundational to entire field of neural networks.

#### Rosenblatt (1958)
**Title:** The Perceptron: A Probabilistic Model for Information Storage and Organization
**Status:** INCLUDED
**PDF:** `rosenblatt-1958.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Historical Significance
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** First learning algorithm for neural networks. Introduced the concept of learning from experience through weight adjustment.

#### Hopfield (1982)
**Title:** Neural Networks and Physical Systems with Emergent Collective Computational Abilities
**Status:** INCLUDED
**PDF:** `hopfield-1982.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Energy-based formulation of neural networks. Influenced later architectures and associative memory concepts relevant to modern retrieval.

#### Rumelhart, Hinton & Williams (1986)
**Title:** Learning Representations by Back-Propagating Errors
**Status:** INCLUDED
**PDF:** `rumelhart-hinton-williams-1986.pdf`
**Criteria Met:** Foundational Impact, Universal Adoption, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Backpropagation remains the fundamental training algorithm for all neural networks. Most influential paper in the collection.

#### Elman (1990)
**Title:** Finding Structure in Time
**Status:** INCLUDED
**PDF:** `elman-1990.pdf`
**Criteria Met:** Foundational Impact, Historical Significance
**Criteria Against:** Superseded by LSTM
**Verdict:** KEEP (marginal)
**Rationale:** Introduced simple recurrent networks for temporal sequences. Historically important bridge to LSTM, though less critical than LSTM itself.

### Excluded Papers

#### Minsky & Papert (1969)
**Title:** Perceptrons
**Status:** EXCLUDED
**Criteria Met:** Historical Significance
**Criteria Against:** Did not advance capabilities; showed limitations
**Verdict:** EXCLUDE
**Rationale:** Important historically for contributing to "AI winter" but critiqued rather than advanced neural networks. Does not meet primary criteria.

---

## Part II: Sequence Models & Word Embeddings (1990-2013)

### Included Papers

#### Hochreiter & Schmidhuber (1997)
**Title:** Long Short-Term Memory
**Status:** INCLUDED
**PDF:** `hochreiter-schmidhuber-1997.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Universal Adoption (until Transformers)
**Criteria Against:** Superseded by Transformers
**Verdict:** KEEP
**Rationale:** Solved the vanishing gradient problem; dominated sequence modeling for 20 years. Essential foundation for understanding sequence processing evolution.

#### Bengio et al. (2003)
**Title:** A Neural Probabilistic Language Model
**Status:** INCLUDED
**PDF:** `bengio-2003.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** First neural language model with learned word embeddings. Conceptual foundation for all LLMs.

#### Glorot & Bengio (2010)
**Title:** Understanding the Difficulty of Training Deep Feedforward Neural Networks
**Status:** INCLUDED
**PDF:** `glorot-bengio-2010.pdf`
**Criteria Met:** Foundational Impact, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Xavier initialization enabled training of deeper networks. Practical technique still in widespread use.

#### Collobert et al. (2011)
**Title:** Natural Language Processing (Almost) from Scratch
**Status:** INCLUDED
**PDF:** `collobert-weston-2011.pdf`
**Criteria Met:** Paradigm Shift, Historical Significance
**Criteria Against:** Superseded by word2vec and transformers
**Verdict:** KEEP
**Rationale:** Demonstrated end-to-end neural NLP without hand-crafted features; bridge between Bengio 2003 and word2vec.

#### Mikolov et al. (2013)
**Title:** Efficient Estimation of Word Representations in Vector Space (word2vec)
**Status:** INCLUDED
**PDF:** `mikolov-2013.pdf`
**Criteria Met:** Foundational Impact, Universal Adoption, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Made word embeddings practical at scale; demonstrated that vector arithmetic could capture semantic relationships. Highly influential.

#### Kingma & Welling (2013)
**Title:** Auto-Encoding Variational Bayes (VAE)
**Status:** INCLUDED
**PDF:** `kingma-welling-2013-vae.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** Less directly applicable to LLMs than other techniques
**Verdict:** KEEP (marginal)
**Rationale:** Foundational for generative modeling; latent representation concepts inform some LLM architectures and applications.

#### Graves (2013)
**Title:** Generating Sequences with Recurrent Neural Networks
**Status:** INCLUDED
**PDF:** `graves-2013.pdf`
**Criteria Met:** Foundational Impact, Historical Significance
**Criteria Against:** Superseded by Transformers
**Verdict:** KEEP
**Rationale:** Demonstrated character-level generation and attention mechanisms for RNNs. Important stepping stone to modern sequence generation.

### Excluded Papers

#### Mikolov et al. (2010)
**Title:** Recurrent Neural Network Based Language Model
**Status:** EXCLUDED
**Criteria Met:** Historical Significance
**Criteria Against:** Superseded by word2vec (2013) and LSTM applications
**Verdict:** EXCLUDE
**Rationale:** Intermediate work from the same author; word2vec is more influential and sufficient for the historical record.

---

## Part III: Deep Learning & Attention (2012-2015)

### Included Papers

#### Krizhevsky et al. (2012)
**Title:** ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
**Status:** INCLUDED
**PDF:** `krizhevsky-2012.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Enabling Breakthrough
**Criteria Against:** Vision-focused, not language
**Verdict:** KEEP
**Rationale:** Triggered the deep learning revolution; demonstrated GPU training, dropout, and ReLU at scale. Essential context for understanding why deep learning succeeded.

#### Srivastava et al. (2014)
**Title:** Dropout: A Simple Way to Prevent Neural Networks from Overfitting
**Status:** INCLUDED
**PDF:** `srivastava-2014-dropout.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Regularization technique used in nearly all neural networks. Essential infrastructure.

#### Simonyan & Zisserman (2014)
**Title:** Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)
**Status:** INCLUDED
**PDF:** `simonyan-zisserman-2014.pdf`
**Criteria Met:** Foundational Impact, Methodological Innovation
**Criteria Against:** Vision-focused
**Verdict:** KEEP (marginal)
**Rationale:** Demonstrated that depth systematically improves performance; influenced architectural thinking that informed Transformers.

#### Kingma & Ba (2014)
**Title:** Adam: A Method for Stochastic Optimization
**Status:** INCLUDED
**PDF:** `kingma-ba-2014.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Default optimizer for most LLM training. Essential infrastructure.

#### Cho et al. (2014)
**Title:** Learning Phrase Representations using RNN Encoder-Decoder (GRU)
**Status:** INCLUDED
**PDF:** `cho-2014.pdf`
**Criteria Met:** Foundational Impact
**Criteria Against:** Superseded by Transformers
**Verdict:** KEEP
**Rationale:** Simplified LSTM architecture; influenced gating mechanisms. Part of sequence modeling evolution.

#### Sutskever et al. (2014)
**Title:** Sequence to Sequence Learning with Neural Networks
**Status:** INCLUDED
**PDF:** `sutskever-2014.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Encoder-decoder framework foundational to all translation and generation systems. Essential.

#### Bahdanau et al. (2014)
**Title:** Neural Machine Translation by Jointly Learning to Align and Translate
**Status:** INCLUDED
**PDF:** `bahdanau-2014.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Universal Adoption
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Invented the attention mechanism. One of the most important papers in the collection.

#### Pennington et al. (2014)
**Title:** GloVe: Global Vectors for Word Representation
**Status:** INCLUDED
**PDF:** `pennington-2014.pdf`
**Criteria Met:** Foundational Impact
**Criteria Against:** Superseded by contextual embeddings
**Verdict:** KEEP (marginal)
**Rationale:** Alternative to word2vec using global co-occurrence statistics. Less critical than word2vec but still influential in embedding research.

#### Luong et al. (2015)
**Title:** Effective Approaches to Attention-based Neural Machine Translation
**Status:** INCLUDED
**PDF:** `luong-2015.pdf`
**Criteria Met:** Foundational Impact
**Criteria Against:** Superseded by self-attention
**Verdict:** KEEP
**Rationale:** Introduced multiplicative attention and local attention variants. Influenced Transformer design.

#### Ioffe & Szegedy (2015)
**Title:** Batch Normalization: Accelerating Deep Network Training
**Status:** INCLUDED
**PDF:** `ioffe-szegedy-2015.pdf`
**Criteria Met:** Enabling Breakthrough, Universal Adoption (for CNNs)
**Criteria Against:** LLMs use LayerNorm instead
**Verdict:** KEEP
**Rationale:** Normalization concept essential; direct precursor to LayerNorm used in Transformers.

#### He et al. (2015)
**Title:** Deep Residual Learning for Image Recognition (ResNet)
**Status:** INCLUDED
**PDF:** `he-2015-resnet.pdf`
**Criteria Met:** Foundational Impact, Universal Adoption, Paradigm Shift
**Criteria Against:** Vision-focused
**Verdict:** KEEP
**Rationale:** Skip/residual connections used in all Transformers. Essential architectural innovation.

#### Sennrich et al. (2015)
**Title:** Neural Machine Translation of Rare Words with Subword Units (BPE)
**Status:** INCLUDED
**PDF:** `sennrich-2015.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** BPE tokenization used by nearly all LLMs. Essential infrastructure.

---

## Part IV: The Transformer Era and Pretraining Revolution (2016-2019)

### Included Papers

#### Ba, Kiros & Hinton (2016)
**Title:** Layer Normalization
**Status:** INCLUDED
**PDF:** `ba-2016-layernorm.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Used in every Transformer architecture. Essential infrastructure.

#### Vaswani et al. (2017)
**Title:** Attention Is All You Need
**Status:** INCLUDED
**PDF:** `vaswani-2017.pdf`
**Criteria Met:** ALL PRIMARY CRITERIA
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** The Transformer architecture. Most important architectural paper since backpropagation. Foundation for all modern LLMs.

#### Shazeer et al. (2017)
**Title:** Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
**Status:** INCLUDED
**PDF:** `shazeer-2017-moe.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Introduced sparsely-gated mixture-of-experts; foundation for Mixtral, DeepSeek-V3, and other sparse models.

#### Schulman et al. (2017)
**Title:** Proximal Policy Optimization Algorithms (PPO)
**Status:** INCLUDED
**PDF:** `schulman-2017-ppo.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** Not language-specific
**Verdict:** KEEP
**Rationale:** PPO is the standard RL algorithm for RLHF. Essential infrastructure for alignment.

#### Christiano et al. (2017)
**Title:** Deep Reinforcement Learning from Human Preferences (RLHF)
**Status:** INCLUDED
**PDF:** `christiano-2017-rlhf.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Introduced learning from human preferences. Foundation for InstructGPT and all aligned LLMs.

#### Peters et al. (2018)
**Title:** Deep Contextualized Word Representations (ELMo)
**Status:** INCLUDED
**PDF:** `peters-2018.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** Superseded by BERT/GPT
**Verdict:** KEEP
**Rationale:** First contextual word embeddings. Established the pretraining paradigm.

#### Devlin et al. (2018)
**Title:** BERT: Pre-training of Deep Bidirectional Transformers
**Status:** INCLUDED
**PDF:** `devlin-2018.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Universal Adoption
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Bidirectional pretraining; massive impact on NLP. Essential.

#### Radford et al. (2018)
**Title:** Improving Language Understanding by Generative Pre-Training (GPT-1)
**Status:** INCLUDED
**PDF:** `radford-2018.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Established autoregressive pretraining. Foundation for the GPT series.

#### Liu et al. (2019)
**Title:** RoBERTa: A Robustly Optimized BERT Pretraining Approach
**Status:** INCLUDED
**PDF:** `liu-2019.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Incremental over BERT
**Verdict:** KEEP (marginal)
**Rationale:** Demonstrated that training recipe matters as much as architecture; influenced subsequent pretraining practices.

#### Raffel et al. (2019)
**Title:** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
**Status:** INCLUDED
**PDF:** `raffel-2019.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Unified text-to-text framework; systematic study of transfer learning approaches.

### Candidate Papers

#### Howard & Ruder (2018)
**Title:** Universal Language Model Fine-tuning for Text Classification (ULMFiT)
**Status:** CANDIDATE
**PDF:** `howard-ruder-2018.pdf` (present but not referenced)
**Criteria Met:** Historical Significance, Paradigm Shift
**Criteria Against:** Superseded by BERT/GPT; transfer learning covered by T5
**Verdict:** CONSIDER ADDING
**Rationale:** First to demonstrate fine-tuning pretrained language models for downstream tasks, before BERT/GPT. Could argue it deserves inclusion as the paper that established the pretrain-then-fine-tune paradigm.

---

## Part V: Emergence and Scale (2019-2020)

### Included Papers

#### Radford et al. (2019)
**Title:** Language Models are Unsupervised Multitask Learners (GPT-2)
**Status:** INCLUDED
**PDF:** `radford-2019.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Revealed Emergence
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Demonstrated zero-shot learning; showed that scaling yields qualitatively new capabilities.

#### Shazeer (2019)
**Title:** Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)
**Status:** INCLUDED
**PDF:** `shazeer-2019-mqa.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Efficiency technique used in most production LLMs for faster inference.

#### Brown et al. (2020)
**Title:** Language Models are Few-Shot Learners (GPT-3)
**Status:** INCLUDED
**PDF:** `brown-2020.pdf`
**Criteria Met:** ALL PRIMARY CRITERIA
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Demonstrated few-shot and in-context learning. Defining paper for the LLM era.

#### Kaplan et al. (2020)
**Title:** Scaling Laws for Neural Language Models
**Status:** INCLUDED
**PDF:** `kaplan-2020.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Methodological Innovation
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Quantified scaling behavior; influenced all subsequent LLM development decisions.

#### Lepikhin et al. (2020)
**Title:** GShard: Scaling Giant Models with Conditional Computation
**Status:** INCLUDED
**PDF:** `lepikhin-2020-gshard.pdf`
**Criteria Met:** Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Demonstrated MoE at scale; enabled trillion-parameter models.

#### Clark et al. (2020)
**Title:** ELECTRA: Pre-training Text Encoders as Discriminators
**Status:** INCLUDED
**PDF:** `clark-2020.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Limited adoption compared to BERT/GPT
**Verdict:** KEEP (marginal)
**Rationale:** Efficient pretraining via replaced token detection; influenced subsequent efficiency work.

#### Shazeer (2020)
**Title:** GLU Variants Improve Transformer
**Status:** INCLUDED
**PDF:** `shazeer-2020.pdf`
**Criteria Met:** Universal Adoption
**Criteria Against:** Incremental improvement
**Verdict:** KEEP
**Rationale:** SwiGLU activation used in LLaMA, PaLM, and most modern LLMs.

#### Lewis et al. (2020)
**Title:** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)
**Status:** INCLUDED
**PDF:** `lewis-2020.pdf`
**Criteria Met:** Foundational Impact, Universal Adoption, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** RAG is standard for knowledge-grounded generation. Combined parametric models with retrieval for improved factual accuracy. Already included in Part V.

### Excluded Papers

#### Choromanski et al. (2020)
**Title:** Rethinking Attention with Performers
**Status:** EXCLUDED
**PDF:** `choromanski-2020.pdf` (present)
**Criteria Met:** Methodological Innovation
**Criteria Against:** Limited real-world adoption; linear attention not dominant
**Verdict:** EXCLUDE
**Rationale:** Interesting theoretical approximation but not adopted in practice. FlashAttention solved the efficiency problem differently and more successfully.

---

## Part VI: Efficiency, Alignment, and Reasoning (2021-2022)

### Included Papers

#### Wei et al. (2021)
**Title:** Finetuned Language Models Are Zero-Shot Learners (FLAN)
**Status:** INCLUDED
**PDF:** `wei-2021-flan.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Instruction tuning became standard practice for aligning LLMs to follow instructions.

#### Hu et al. (2021)
**Title:** LoRA: Low-Rank Adaptation of Large Language Models
**Status:** INCLUDED
**PDF:** `hu-2021.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Standard for parameter-efficient fine-tuning. Essential infrastructure.

#### Su et al. (2021)
**Title:** RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
**Status:** INCLUDED
**PDF:** `su-2021.pdf`
**Criteria Met:** Universal Adoption
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** RoPE position encoding used in LLaMA, Mistral, and most modern LLMs. Essential.

#### Dao et al. (2022)
**Title:** FlashAttention: Fast and Memory-Efficient Exact Attention
**Status:** INCLUDED
**PDF:** `dao-2022.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** IO-aware attention used in all production LLMs. Essential infrastructure for efficient training and inference.

#### Ouyang et al. (2022)
**Title:** Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
**Status:** INCLUDED
**PDF:** `ouyang-2022.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Universal Adoption
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Established RLHF for LLM alignment. Every major LLM chatbot uses this or closely related techniques.

#### Wei et al. (2022)
**Title:** Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**Status:** INCLUDED
**PDF:** `wei-2022.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift, Revealed Emergence
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Enabled multi-step reasoning through prompting; foundation for o1-style reasoning systems.

#### Bai et al. (2022)
**Title:** Constitutional AI: Harmlessness from AI Feedback
**Status:** INCLUDED
**PDF:** `bai-2022.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** Primarily Anthropic-specific initially
**Verdict:** KEEP
**Rationale:** Introduced self-critique and principle-based training. Influenced industry-wide safety practices. See also: [Claude's Constitution](https://www.anthropic.com/constitution) for practical implementation.

#### Yao et al. (2022)
**Title:** ReAct: Synergizing Reasoning and Acting in Language Models
**Status:** INCLUDED
**PDF:** `yao-2022.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Synergized reasoning and acting for tool use. Foundation for agent architectures.

#### Hoffmann et al. (2022)
**Title:** Training Compute-Optimal Large Language Models (Chinchilla)
**Status:** INCLUDED
**PDF:** `hoffmann-2022.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Compute-optimal training ratios changed LLM development strategy industry-wide.

#### Chowdhery et al. (2022)
**Title:** PaLM: Scaling Language Modeling with Pathways
**Status:** INCLUDED
**PDF:** `chowdhery-2022.pdf`
**Criteria Met:** Foundational Impact, Revealed Emergence
**Criteria Against:** System report rather than technique paper
**Verdict:** KEEP (marginal)
**Rationale:** Demonstrated emergence at 540B scale; pathway-based distributed architecture. Borderline between main body and Appendix C but emergence findings justify main body inclusion.

### Excluded Papers

#### Press et al. (2021)
**Title:** Train Short, Test Long: Attention with Linear Biases (ALiBi)
**Status:** EXCLUDED
**PDF:** `press-2021.pdf` (present)
**Criteria Met:** Foundational Impact
**Criteria Against:** Less adopted than RoPE
**Verdict:** EXCLUDE
**Rationale:** RoPE won the position encoding competition. Including both is redundant; RoPE is more widely used.

#### Borgeaud et al. (2021)
**Title:** Improving Language Models by Retrieving from Trillions of Tokens (RETRO)
**Status:** EXCLUDED
**PDF:** `borgeaud-2021.pdf` (present)
**Criteria Met:** Foundational Impact
**Criteria Against:** Limited adoption; RAG more prevalent for retrieval
**Verdict:** EXCLUDE
**Rationale:** Retrieval-augmented training is interesting but not widely adopted. RAG at inference time is more prevalent.

#### Fedus et al. (2022)
**Title:** Switch Transformers: Scaling to Trillion Parameter Models
**Status:** EXCLUDED
**PDF:** `fedus-2022.pdf` (present)
**Criteria Met:** Foundational Impact
**Criteria Against:** Superseded by later MoE work
**Verdict:** EXCLUDE (reconsidered)
**Rationale:** First successful sparse MoE at trillion-parameter scale, but Shazeer 2017 is more foundational and Mixtral is more influential as the open MoE breakthrough.

---

## Part VII: Open LLMs and Modern Frontier (2023-2024)

### Included Papers

#### Touvron et al. (2023)
**Title:** LLaMA: Open and Efficient Foundation Language Models
**Status:** INCLUDED
**PDF:** `touvron-2023.pdf`
**Criteria Met:** Paradigm Shift, Open Ecosystem Impact
**Criteria Against:** System report
**Verdict:** KEEP
**Rationale:** Democratized LLM access; enabled entire open-source ecosystem.

#### Touvron et al. (2023)
**Title:** Llama 2: Open Foundation and Fine-Tuned Chat Models
**Status:** INCLUDED
**PDF:** `touvron-2023-llama2.pdf`
**Criteria Met:** Open Ecosystem Impact
**Criteria Against:** Incremental over LLaMA 1
**Verdict:** KEEP (marginal)
**Rationale:** First major open model with RLHF training; disclosed safety methodology. Questionable if both LLaMA 1 and 2 needed.

#### Rafailov et al. (2023)
**Title:** Direct Preference Optimization (DPO)
**Status:** INCLUDED
**PDF:** `rafailov-2023-dpo.pdf`
**Criteria Met:** Foundational Impact, Universal Adoption
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Simplified alignment without explicit reward modeling or RL. Widely adopted alternative to PPO-based RLHF.

#### Lee et al. (2023)
**Title:** RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
**Status:** INCLUDED
**PDF:** `lee-2023-rlaif.pdf`
**Criteria Met:** Foundational Impact, Paradigm Shift
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** AI-generated preferences enable scaling Constitutional AI principles beyond human labeling capacity.

#### Dettmers et al. (2023)
**Title:** QLoRA: Efficient Finetuning of Quantized LLMs
**Status:** INCLUDED
**PDF:** `dettmers-2023.pdf`
**Criteria Met:** Universal Adoption, Enabling Breakthrough
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Enabled fine-tuning of 65B+ models on consumer hardware. Essential for democratization.

#### Ainslie et al. (2023)
**Title:** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
**Status:** INCLUDED
**PDF:** `ainslie-2023-gqa.pdf`
**Criteria Met:** Universal Adoption
**Criteria Against:** Incremental efficiency improvement
**Verdict:** KEEP
**Rationale:** Grouped-query attention used in LLaMA 2, Mistral, and most modern open LLMs. Standard technique.

#### Jiang et al. (2024)
**Title:** Mixtral of Experts
**Status:** INCLUDED
**PDF:** `mistral-2024.pdf`
**Criteria Met:** Paradigm Shift, Open Ecosystem Impact
**Criteria Against:** None
**Verdict:** KEEP
**Rationale:** Validated sparse MoE for open models; demonstrated that MoE can match larger dense models with less compute.

#### Yuan et al. (2024)
**Title:** Self-Rewarding Language Models
**Status:** INCLUDED
**PDF:** `yuan-2024-self-rewarding.pdf`
**Criteria Met:** Foundational Impact
**Criteria Against:** Too recent to assess full impact
**Verdict:** KEEP (marginal)
**Rationale:** Extension of Constitutional AI principles to self-improvement. Part of alignment methodology evolution.

#### Zhu et al. (2024)
**Title:** DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model
**Status:** INCLUDED
**PDF:** `zhu-2024-deepseek-v2.pdf`
**Criteria Met:** Methodological Innovation (Multi-Head Latent Attention)
**Criteria Against:** Company-specific architecture details
**Verdict:** KEEP (marginal)
**Rationale:** Multi-Head Latent Attention is a novel contribution. Borderline between main body and Appendix C.

#### Snell et al. (2024)
**Title:** Scaling LLM Test-Time Compute Optimally
**Status:** INCLUDED
**PDF:** `snell-2024-test-time-compute.pdf`
**Criteria Met:** Paradigm Shift, Revealed Emergence
**Criteria Against:** Too recent
**Verdict:** KEEP
**Rationale:** Foundation for o1-style inference-time scaling. Potentially paradigm-shifting if test-time compute scaling becomes dominant approach.

### Papers to Remove (Moderate Policy)

#### Rozière et al. (2023)
**Title:** Code Llama: Open Foundation Models for Code
**Status:** INCLUDED → REMOVE
**PDF:** `roziere-2023.pdf`
**Criteria Met:** Open Ecosystem Impact only
**Criteria Against:** Specialized application variant
**Verdict:** REMOVE
**Rationale:** Application of LLaMA architecture to code. Influential for code generation but not foundational; a specialized variant rather than architectural innovation.

#### Liu et al. (2023)
**Title:** Ring Attention with Blockwise Transformers for Near-Infinite Context
**Status:** INCLUDED → REMOVE
**PDF:** `liu-2023-ring-attention.pdf`
**Criteria Met:** Enabling Breakthrough only
**Criteria Against:** Niche; context length is one of many solutions
**Verdict:** REMOVE
**Rationale:** Context extension technique with limited universal adoption compared to other approaches.

#### Shao et al. (2024)
**Title:** DeepSeekMath: Pushing the Limits of Mathematical Reasoning
**Status:** INCLUDED → REMOVE
**PDF:** `shao-2024-deepseekmath.pdf`
**Criteria Met:** Methodological Innovation only
**Criteria Against:** Domain-specific; GRPO not widely adopted
**Verdict:** REMOVE
**Rationale:** Interesting for GRPO methodology but too specialized for main body. Math-specific paper rather than general LLM foundation.

### Papers to Move

#### Dubey et al. (2024)
**Title:** The Llama 3 Herd of Models
**Status:** INCLUDED → MOVE TO APPENDIX C
**PDF:** `touvron-2024-llama3.pdf`
**Criteria Met:** Open Ecosystem Impact
**Criteria Against:** Third LLaMA paper; system report character
**Verdict:** MOVE to Appendix C
**Rationale:** Having three LLaMA papers (1, 2, 3) in Part VII is excessive. Llama 3 is more of a system report showing technique integration; better fit for Appendix C.

#### Munkhdalai et al. (2024)
**Title:** Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
**Status:** INCLUDED → MOVE TO APPENDIX A
**PDF:** `munkhdalai-2024-infini-attention.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Too recent to assess adoption
**Verdict:** MOVE to Appendix A
**Rationale:** Promising infinite context approach but adoption unclear. Better fit for Appendix A (Emerging Results).

---

## Appendix A: Emerging Results (2023-2025)

### Included Papers

#### Gu & Dao (2023)
**Title:** Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Status:** INCLUDED
**PDF:** `gu-2023.pdf`
**Criteria Met:** Paradigm Shift, Methodological Innovation
**Criteria Against:** Adoption unclear; SSMs still nascent
**Verdict:** KEEP
**Rationale:** Leading alternative architecture to Transformers. Important to include even if not yet dominant.

#### Cunningham et al. (2023)
**Title:** Sparse Autoencoders Find Highly Interpretable Features
**Status:** INCLUDED
**PDF:** `cunningham-2023.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Interpretability is niche
**Verdict:** KEEP
**Rationale:** Important for mechanistic interpretability and safety research.

#### Bai et al. (2023)
**Title:** Safe RLHF: Safe Reinforcement Learning from Human Feedback
**Status:** INCLUDED
**PDF:** `bai-2023.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Niche safety application
**Verdict:** KEEP
**Rationale:** Extends RLHF with explicit safety constraints. Important for alignment research.

#### Frantar et al. (2024)
**Title:** Scaling Laws for Fine-Grained Mixture of Experts
**Status:** INCLUDED
**PDF:** `frantar-2024.pdf`
**Criteria Met:** Methodological Innovation
**Criteria Against:** Too recent to assess
**Verdict:** KEEP
**Rationale:** Extends scaling laws to MoE architectures; important for architectural decisions.

### Papers to Remove

#### Casper et al. (2024)
**Title:** Open Problems and Fundamental Limitations of RLHF
**Status:** INCLUDED → REMOVE
**PDF:** `casper-2024.pdf`
**Criteria Met:** None (survey paper)
**Criteria Against:** Survey rather than primary research contribution
**Verdict:** REMOVE
**Rationale:** Survey paper documenting challenges rather than introducing techniques. Does not meet inclusion criteria for foundational papers.

### Papers to Add

#### Munkhdalai et al. (2024) - Infini-attention
**Status:** MOVE FROM Part VII
**Rationale:** Too recent to assess adoption; better fit for emerging results.

---

## Appendix B: Foundations of Agents (2022-2025)

### Included Papers

#### Liang et al. (2022)
**Title:** Code as Policies: Language Model Programs for Embodied Control
**Status:** INCLUDED
**PDF:** `liang-2022.pdf`
**Verdict:** KEEP
**Rationale:** Demonstrated code generation for robotics control.

#### Schick et al. (2023)
**Title:** Toolformer: Language Models Can Teach Themselves to Use Tools
**Status:** INCLUDED
**PDF:** `schick-2023.pdf`
**Verdict:** KEEP
**Rationale:** Self-supervised tool learning.

#### Yao et al. (2023)
**Title:** Tree of Thoughts: Deliberate Problem Solving with Large Language Models
**Status:** INCLUDED
**PDF:** `yao-2023.pdf`
**Verdict:** KEEP
**Rationale:** Structured reasoning with search.

#### Wang et al. (2024)
**Title:** Executable Code Actions Elicit Better LLM Agents (CodeAct)
**Status:** INCLUDED
**PDF:** `wang-2024.pdf`
**Verdict:** KEEP
**Rationale:** Code as unified action space for agents.

### Missing Papers

#### Shinn et al. (2023)
**Title:** Reflexion: Language Agents with Verbal Reinforcement Learning
**Status:** REFERENCED BUT MISSING
**PDF:** Not present - needs to be added
**Verdict:** ADD PDF
**Rationale:** Referenced in Appendix B intro but PDF not in repository.

---

## Appendix C: System Reports & Production Breakthroughs (2023-2025)

### Included Papers

#### OpenAI (2023) - GPT-4
**Title:** GPT-4 Technical Report
**Status:** INCLUDED
**PDF:** `openai-2023.pdf`
**Verdict:** KEEP
**Rationale:** Major milestone documenting multimodal frontier model.

#### Anthropic (2024) - Claude 3
**Title:** The Claude 3 Model Family
**Status:** INCLUDED
**PDF:** `anthropic-2024.pdf`
**Verdict:** KEEP
**Rationale:** Constitutional AI applied to production systems at scale.

#### DeepMind (2024) - Gemini 1.5
**Title:** Gemini 1.5: Unlocking Multimodal Understanding
**Status:** INCLUDED
**PDF:** `deepmind-2024.pdf`
**Verdict:** KEEP
**Rationale:** Multimodal model with million-token context capability.

#### Qwen Team (2024)
**Title:** Qwen2.5 Technical Report
**Status:** INCLUDED
**PDF:** `qwen-2024-2.5.pdf`
**Verdict:** KEEP (marginal)
**Rationale:** Strong multilingual open model series; less distinctive than other reports.

#### DeepSeek-AI (2024) - DeepSeek-V3
**Title:** DeepSeek-V3 Technical Report
**Status:** INCLUDED
**PDF:** `deepseek-2024-v3.pdf`
**Verdict:** KEEP
**Rationale:** Cost-efficient training; validates MoE at 671B scale.

#### OpenAI (2024) - o1
**Title:** OpenAI o1 System Card
**Status:** INCLUDED
**PDF:** `openai-2024-o1-system-card.pdf`
**Verdict:** KEEP
**Rationale:** Documents inference-time reasoning breakthrough.

#### Kumar et al. (2025)
**Title:** LLM Post-Training: A Deep Dive into Reasoning and Large Language Models
**Status:** INCLUDED
**PDF:** `kumar-2025-post-training-survey.pdf`
**Verdict:** KEEP
**Rationale:** Comprehensive survey of post-training techniques; valuable reference.

### Papers to Remove

#### OpenAI (2025) - o3 Competitive Programming
**Title:** Competitive Programming with Large Reasoning Models
**Status:** INCLUDED → REMOVE
**PDF:** `openai-2025-o3-competitive-programming.pdf`
**Criteria Against:** Very narrow focus (competitive programming only)
**Verdict:** REMOVE
**Rationale:** Too narrow; focuses only on competitive programming benchmarks. The o1 system card already covers inference-time reasoning.

### Papers to Add

#### Dubey et al. (2024) - Llama 3
**Status:** MOVE FROM Part VII
**PDF:** `touvron-2024-llama3.pdf`
**Rationale:** Better characterized as system report showing technique integration.

---

## Orphaned PDFs Requiring Resolution

These PDFs are present in the repository but not referenced in any section:

| PDF File | Status | Recommendation |
|----------|--------|----------------|
| `howard-ruder-2018.pdf` | CANDIDATE | Consider adding to Part IV |
| `lewis-2020.pdf` | CANDIDATE | Add to Part V (RAG) |
| `choromanski-2020.pdf` | EXCLUDED | Delete or archive |
| `borgeaud-2021.pdf` | EXCLUDED | Delete or archive |
| `fedus-2022.pdf` | EXCLUDED | Delete or archive |
| `press-2021.pdf` | EXCLUDED | Delete or archive |
| `gu-dao-2023-mamba.pdf` | DUPLICATE | Delete (keep `gu-2023.pdf`) |
| `openai-2024-o1.pdf` | DUPLICATE | Delete (keep `openai-2024-o1-system-card.pdf`) |
| `chen-2025.pdf` | UNKNOWN | Identify content or delete |
| `erdogan-2025.pdf` | UNKNOWN | Identify content or delete |
| `jiang-2025.pdf` | UNKNOWN | Identify content or delete |
| `li-2025-rlu.pdf` | UNKNOWN | Referenced in CLAUDE.md; evaluate for Appendix A |
| `li-2025-smoe.pdf` | UNKNOWN | Referenced in CLAUDE.md; evaluate for Appendix A |
| `liu-2025.pdf` | UNKNOWN | Identify content or delete |
| `wang-2025.pdf` | UNKNOWN | Identify content or delete |
| `yan-2025.pdf` | UNKNOWN | Identify content or delete |
| `zhang-2025.pdf` | UNKNOWN | Identify content or delete |
| `zhao-2025.pdf` | UNKNOWN | Referenced in CLAUDE.md (RoPE extrapolation); evaluate |

---

## Non-Paper Documents

### Anthropic Claude Constitution (2025)
**URL:** https://www.anthropic.com/constitution
**Format:** Web + PDF/ePub (CC0)
**Type:** Specification document, not research paper
**Decision:** DO NOT ADD as standalone paper
**Alternative:** Reference in Constitutional AI (Bai 2022) summary as "the publicly released implementation of these principles"
**Rationale:** Documents how Constitutional AI is applied in practice but doesn't introduce new research methodology.

---

*Last updated: January 2026*
