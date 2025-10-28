# LaTeX Restructuring Plan
## Integrating RL, Constitutional AI, and Recent Breakthroughs

---

## Current Structure Analysis

**Current Parts (main.tex):**
- Part I: Neural Beginnings (1943–1990) - 4 papers
- Part II: Sequence Models & Word Embeddings (1990–2013) - 4 papers
- Part III: Deep Learning Revolution (2012–2016) - 4 papers ⚠️ **OVERLAPS with Part IV**
- Part IV: Attention & Seq2Seq (2014–2016) - 4 papers ⚠️ **OVERLAPS with Part III**
- Part V: Transformer Era (2017–2019) - 5 papers
- Part VI: Emergence & Scale (2019–2020) - 5 papers
- Part VII: Efficiency, Alignment, Reasoning (2021–2022) - ~10 papers
- Part VIII: Open LLMs & Modern Frontier (2023–2024) - ~10 papers
- Appendix A: Emerging Results (2025) - ~5 papers
- Appendix B: Foundations of Agents (2022–2025) - ~6 papers

**Problems with Current Structure:**
1. Parts III and IV have overlapping date ranges (2012-2016 vs 2014-2016)
2. Missing RL foundations entirely
3. Missing RLHF/Constitutional AI evolution
4. Missing recent 2024-2025 breakthroughs
5. System reports mixed with research papers

---

## Proposed New Structure

### **Part 0 – Probabilistic & Mathematical Foundations (1906–1943)** ✨ NEW
**Papers to add:**
- [ ] Markov (1906-1913) - Markov Chains (need to find source)
- [x] McCulloch & Pitts (1943) - Move from Part I

**Files to create:**
- `content/part0-chapter.tex`
- `content/part0-intro.tex` (use from `section_introductions.tex`)

---

### **Part I – Neural Beginnings & RL Foundations (1943–1990)** 🔄 UPDATED
**Current papers (keep):**
- [x] Rosenblatt (1958) - Perceptron
- [x] Hopfield (1982) - Neural Networks
- [x] Rumelhart et al. (1986) - Backpropagation

**Papers to add:**
- [ ] Bellman (1957) - Dynamic Programming ⚠️ **NEED TO FIND**
- [ ] Watkins (1989/1992) - Q-Learning ⚠️ **NEED TO FIND**

**Files to update:**
- `content/part1-chapter.tex` - Update intro from `section_introductions.tex`
- Add Bellman and Watkins PDFs

---

### **Part II – Sequence Models & Word Embeddings (1990–2013)** ✅ NO CHANGES
**Current papers (all keep):**
- [x] Hochreiter & Schmidhuber (1997) - LSTM
- [x] Bengio et al. (2003) - Neural Probabilistic LM
- [x] Mikolov et al. (2013) - Word2Vec
- [x] Graves (2013) - Generating Sequences

---

### **Part III – Deep Learning & Deep RL (2012–2016)** 🔄 MERGE & REORGANIZE
**Strategy:** Merge current Parts III and IV, add Deep RL papers

**Papers to keep from current Part III:**
- [x] Krizhevsky et al. (2012) - AlexNet
- [x] Simonyan & Zisserman (2014) - VGGNet
- [x] Ioffe & Szegedy (2015) - Batch Normalization
- [x] He et al. (2015) - ResNet

**Papers to keep from current Part IV:**
- [x] Sutskever et al. (2014) - Seq2Seq
- [x] Bahdanau et al. (2014) - Attention for NMT
- [x] Sennrich et al. (2015) - Subword Units
- [x] Kingma & Ba (2014) - Adam

**Papers to add (Deep RL):**
- [x] **Mnih et al. (2015) - DQN** ✅ Downloaded: `pdfs/mnih-2015-dqn.pdf`
- [x] **Mnih et al. (2016) - A3C** ✅ Downloaded: `pdfs/mnih-2016-a3c.pdf`

**Files to create/update:**
- Merge `content/part3-deeplearning-chapter.tex` and `content/part4-chapter.tex`
- Create new unified `content/part3-chapter.tex`

---

### **Part IV – Transformers, RLHF & Pretraining (2017–2019)** 🔄 UPDATED
**Current papers from Part V (keep):**
- [x] Vaswani et al. (2017) - Attention Is All You Need
- [x] Peters et al. (2018) - ELMo
- [x] Devlin et al. (2018) - BERT
- [x] Radford et al. (2018) - GPT-1
- [x] Raffel et al. (2019) - T5

**Papers to add (RLHF & Deep RL):**
- [x] **Schulman et al. (2017) - PPO** ✅ Downloaded: `pdfs/schulman-2017-ppo.pdf`
- [x] **Christiano et al. (2017) - RLHF** ✅ Downloaded: `pdfs/christiano-2017-rlhf.pdf`
- [x] **Espeholt et al. (2018) - IMPALA** ✅ Downloaded: `pdfs/espeholt-2018-impala.pdf`

**Files to update:**
- Update from current `content/part5-chapter.tex`
- Replace intro with version from `section_introductions.tex`

---

### **Part V – Emergence and Scale (2019–2020)** ✅ MINOR CHANGES
**Current papers from Part VI (keep most):**
- [x] Radford et al. (2019) - GPT-2
- [x] Brown et al. (2020) - GPT-3
- [x] Kaplan et al. (2020) - Scaling Laws
- [ ] ~~Choromanski et al. (2020) - Performers~~ (consider removing, limited adoption)
- [ ] ~~Lewis et al. (2020) - RAG~~ (consider removing, application not foundational)

**Files to update:**
- Rename from `content/part6-chapter.tex` to `content/part5-chapter.tex`

---

### **Part VI – Efficiency, Alignment & Reasoning (2021–2022)** ✅ KEEP MOSTLY AS-IS
**Current papers from Part VII (keep):**
- [x] Hu et al. (2021) - LoRA
- [x] Su et al. (2021) - RoFormer (RoPE)
- [x] Press et al. (2021) - ALiBi
- [x] Borgeaud et al. (2021) - RETRO
- [x] Fedus et al. (2022) - Switch Transformers
- [x] Dao et al. (2022) - FlashAttention
- [x] Ouyang et al. (2022) - InstructGPT
- [x] Bai et al. (2022) - Constitutional AI
- [x] Wei et al. (2022) - Chain-of-Thought
- [x] Yao et al. (2022) - ReAct
- [x] Hoffmann et al. (2022) - Chinchilla

**Files to update:**
- Rename from `content/part7-chapter.tex` to `content/part6-chapter.tex`

---

### **Part VII – Open Models & Advanced Alignment (2023–2024)** 🔄 MAJOR UPDATES
**Current papers from Part VIII (keep):**
- [x] Touvron et al. (2023) - LLaMA
- [x] Dettmers et al. (2023) - QLoRA
- [x] Mistral AI (2024) - Mixtral
- [ ] ~~OpenAI (2023) - GPT-4~~ → **Move to Appendix B**
- [ ] ~~Anthropic (2024) - Claude 3~~ → **Move to Appendix B**
- [ ] ~~DeepMind (2024) - Gemini 1.5~~ → **Move to Appendix B**

**Papers to add:**
- [x] **Rafailov et al. (2023) - DPO** ✅ Downloaded: `pdfs/rafailov-2023-dpo.pdf`
- [x] **Lee et al. (2023) - RLAIF** ✅ Downloaded: `pdfs/lee-2023-rlaif.pdf`
- [x] **Liu et al. (2023) - Ring Attention** ✅ Downloaded: `pdfs/liu-2023-ring-attention.pdf`
- [x] **Munkhdalai et al. (2024) - Infini-attention** ✅ Downloaded: `pdfs/munkhdalai-2024-infini-attention.pdf`
- [x] **Yuan et al. (2024) - Self-Rewarding LMs** ✅ Downloaded: `pdfs/yuan-2024-self-rewarding.pdf`
- [x] **Touvron et al. (2024) - Llama 3** ✅ Downloaded: `pdfs/touvron-2024-llama3.pdf`
- [x] **Shao et al. (2024) - DeepSeekMath** ✅ Downloaded: `pdfs/shao-2024-deepseekmath.pdf`
- [x] **Zhu et al. (2024) - DeepSeek-V2** ✅ Downloaded: `pdfs/zhu-2024-deepseek-v2.pdf`
- [x] **Snell et al. (2024) - Test-Time Compute** ✅ Downloaded: `pdfs/snell-2024-test-time-compute.pdf`

**Files to update:**
- Update from `content/part8-chapter.tex`
- Replace intro with version from `section_introductions.tex`

---

### **Appendix A – Emerging Research (2024-2025)** ✅ KEEP MOSTLY AS-IS
**Current papers (from existing Appendix A):**
- Review and keep relevant emerging research papers

---

### **Appendix B – System Reports & Production Systems (2023-2025)** ✨ NEW
**Strategy:** Move system cards from main body, add new ones

**Papers to move from Part VIII:**
- [ ] OpenAI (2023) - GPT-4 Technical Report (already in pdfs/)
- [ ] Anthropic (2024) - Claude 3 Model Family (already in pdfs/)
- [ ] DeepMind (2024) - Gemini 1.5 Technical Report (already in pdfs/)

**Papers to add:**
- [x] **DeepSeek-V3 (2024)** ✅ Downloaded: `pdfs/deepseek-2024-v3.pdf`
- [x] **Qwen2.5 (2024)** ✅ Downloaded: `pdfs/qwen-2024-2.5.pdf`
- [x] **OpenAI o1 System Card (2024)** ✅ Downloaded: `pdfs/openai-2024-o1-system-card.pdf`
- [x] **OpenAI o3 (2025)** ✅ Downloaded: `pdfs/openai-2025-o3-competitive-programming.pdf`
- [x] **Kumar et al. (2025) - Post-Training Survey** ✅ Downloaded: `pdfs/kumar-2025-post-training-survey.pdf`

**Files to create:**
- `content/appendix-c-chapter.tex` (or repurpose existing Appendix B)

---

## Execution Plan

### Phase 1: Setup & Part 0
1. Create `content/part0-chapter.tex` with Markov + McCulloch & Pitts
2. Update `main.tex` to add Part 0

### Phase 2: Update Part I (RL Foundations)
1. Find/acquire Bellman and Watkins papers
2. Update `content/part1-chapter.tex` with new intro and RL papers

### Phase 3: Merge & Reorganize Part III
1. Merge Parts III and IV into unified Part III
2. Add DQN and A3C papers
3. Create new `content/part3-chapter.tex`

### Phase 4: Update Part IV (RLHF)
1. Add PPO, RLHF, IMPALA papers to current Part V
2. Update intro with RLHF context
3. Rename to Part IV

### Phase 5: Renumber Parts V-VI
1. Rename Part VI → Part V
2. Rename Part VII → Part VI
3. Update all cross-references

### Phase 6: Major Update to Part VII
1. Update Part VIII → Part VII
2. Add all new alignment & architecture papers (DPO, RLAIF, Ring Attention, etc.)
3. Move system reports to new Appendix

### Phase 7: Create Appendix B
1. Repurpose old Appendix B or create Appendix C
2. Add all system reports
3. Update table of contents

### Phase 8: Update Main Files
1. Update `main.tex` with new part structure
2. Update `main-mobile.tex`
3. Update `main-ipad.tex`
4. Update prologue/epilogue if needed

### Phase 9: Build & Test
1. Run `make clean`
2. Run `make all`
3. Verify PDF generation
4. Check table of contents
5. Spot-check paper inclusions

---

## Files Still Needed

⚠️ **Critical Missing Papers:**
1. **Bellman (1957)** - Dynamic Programming
   - Book, not paper - need to find:
     - Key chapter/excerpts, OR
     - Bellman's seminal DP paper, OR
     - Alternative authoritative source

2. **Watkins (1989/1992)** - Q-Learning
   - PhD thesis (1989) - may be hard to find
   - Journal paper (1992): https://link.springer.com/article/10.1007/BF00992698
   - Need accessible PDF

3. **Markov (1906-1913)** - Markov Chains
   - Historical work in Russian
   - Need: English translation or authoritative historical survey

---

## Risk Assessment

**Low Risk:**
- Adding new papers to existing parts
- Creating Part 0
- Creating Appendix B for system reports

**Medium Risk:**
- Merging Parts III and IV
- Renumbering Parts V-VII
- Updating cross-references

**High Risk:**
- LaTeX compilation errors due to missing files
- Broken hyperlinks/labels
- PDF size exceeding reasonable limits

---

## Estimated Timeline

- **Phase 1-2**: 1-2 hours (Part 0 and Part I)
- **Phase 3-4**: 2-3 hours (Merge III/IV, update with RLHF)
- **Phase 5-6**: 2-3 hours (Renumber and update Part VII)
- **Phase 7**: 1 hour (Create Appendix B)
- **Phase 8-9**: 1-2 hours (Build and test)

**Total**: 7-11 hours of work

---

## Recommendation

Given the scope and complexity, I recommend:

**Option A: Full Restructuring (Recommended)**
- Implement complete plan above
- Results in clean, logical structure
- Proper historical progression
- ~7-11 hours of work

**Option B: Incremental Approach**
- Start with just adding new papers to existing structure
- Don't merge/renumber parts yet
- Quick win, but leaves structural issues
- ~3-4 hours of work
- Can restructure later

**Option C: Hybrid Approach**
- Add all new papers first (Phases 1, 2, 6, 7)
- Defer merging Parts III/IV
- Accept some overlap in current structure
- ~4-6 hours of work

Which approach would you like me to take?
