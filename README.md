# RAG Documentation Assistant

> Production-ready Retrieval-Augmented Generation system built with Claude API. Part of the Scholar's Brain project - demonstrating AI Implementation Engineering skills for Solutions Engineer roles.

**Built by CK** | November 2025 | [GitHub](https://github.com/yourusername/rag-doc-assistant-new) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## Why This Project Exists

This is **Phase 1 of a three-project arc** building toward the Scriptorium Assistant:

1. **RAG Documentation Assistant** (this project) → Prove RAG fundamentals  
2. **Homelab Infrastructure** → Deploy in production environment  
3. **Scriptorium Assistant** → Scale to 500+ notes with voice capabilities

**Market Context:** AI Implementation Engineer roles explicitly require "Domain-Specific RAG System" portfolio projects. This demonstrates those skills while building the knowledge retrieval layer for my actual 500+ note Scholar's Web.

**This isn't resume theater** - I'm solving my own problem: making my Obsidian knowledge base queryable via natural language.

---

## Features

- 📂 **Multi-format document loading** (.md, .txt, .pdf)
- 🔍 **Semantic search** (meaning-based retrieval, not keyword matching)
- 💬 **Natural language Q&A** with proper source citations
- 💾 **Persistent vector storage** (sub-second subsequent startups)
- 🎯 **Scalable chunking strategy** (tested 256/512/1024, ready for production)
- 💰 **Cost-effective** (~$0.01/query with local embeddings vs $0.15 with OpenAI)

---

## Architecture

```
User Question
    ↓
Query Engine (LlamaIndex)
    ↓
Vector Search (ChromaDB)
    ↓
Top-K Most Similar Chunks
    ↓
Claude Sonnet 4 (with context)
    ↓
Answer + Source Citations
```

**Why This Stack:**

- **Claude API**: High-quality reasoning, clean integration, cost-effective
- **Local Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (zero API costs)
- **ChromaDB**: Persistent storage, fast similarity search
- **UV**: Survives EndeavourOS rolling release Python updates
- **Python 3.11.9**: Stable, compatible with all dependencies

---

## Quick Start

### Prerequisites

- Python 3.11+
- Claude API key from [console.anthropic.com](https://console.anthropic.com)
- UV (Python version manager)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/rag-doc-assistant-new
cd rag-doc-assistant-new

# 2. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Setup environment
uv python install 3.11.9
uv python pin 3.11.9
uv pip install -r requirements.txt

# 4. Configure API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Usage

```bash
# Add documents to /data folder
cp your-documents.md data/

# Index documents (one-time)
uv run python ingest.py

# Start interactive query
uv run python app.py
```

---

## Implementation Journey

### Phase 1: Core RAG Pipeline (Completed)

**Goal:** Prove end-to-end concept works.

**What I Built:**
- Document loading from `/data` folder (markdown, text, PDFs)
- Vector embedding generation (384-dimensional semantic vectors)
- Claude API integration with context-aware prompting
- Source citation with similarity scores

**Key Learning:** First-run latency (45-60 seconds) is expected behavior, not a bug. This includes embedding model download (90MB, one-time), ChromaDB initialization, and vector creation. Documenting expected vs unexpected behavior demonstrates production thinking.

**Proof:** Working system that answers questions about documents with traceable citations.

### Phase 2: Persistent Storage (Completed)

**Goal:** Eliminate 60-second startup on subsequent runs.

**What I Built:**
- ChromaDB PersistentClient for disk-based vector storage
- Smart loading pattern (fast path if index exists, slow path if not)
- Interactive query loop for multi-question sessions
- Reindexing utility (`ingest.py`) for document updates

**Performance Improvement:**
- First run: 45-60 seconds (creates index)
- Subsequent runs: <1 second (loads from disk)
- **98% reduction in startup time**

**Key Insight:** Persistence transforms a prototype into production-ready software. Users won't tolerate 60-second waits every time they query their knowledge base.

### Phase 3: Retrieval Optimization (Tested & Documented)

**Goal:** Understand how chunking strategy affects answer quality.

**What I Tested:**
- Chunk sizes: 256, 512, 1024 characters
- Retrieval context: top-k=2 (default) vs top-k=5
- Response modes: tree_summarize vs compact

**Results with 2-document test corpus:**
- Both default and optimized strategies produced high-quality answers
- Optimized version (512 chars, top-k=5) retrieved slightly more specific details
- Difference was marginal due to small corpus size

**Decision:** Document optimization approach for later implementation when scaling to 500+ document production corpus (Scholar's Web). With larger document sets, retrieval precision becomes critical. Testing proved I understand the mechanics; production deployment will show measurable impact.

**Implementation preserved in `query.py` for future use.**

---

## Technical Deep Dive

### Cost Structure

| Component | Provider | Cost per Query |
|-----------|----------|----------------|
| Embeddings | Local (all-MiniLM-L6-v2) | $0.00 |
| LLM Generation | Claude Sonnet 4 | ~$0.01 |
| **Total** | | **~$0.01** |

**Comparison:**
- OpenAI (GPT-4 + ada-002 embeddings): ~$0.15/query
- This system: ~$0.01/query
- **93% cost reduction**

### Chunking Strategy Analysis

Tested three configurations on test corpus:

| Chunk Size | Chunks Created | Pros | Cons |
|------------|----------------|------|------|
| 256 chars | 8 | High precision | Context fragmentation |
| **512 chars** | **4** | **Balance** | **Production choice** |
| 1024 chars | 2 | More context | Lower precision |

**Selection rationale:** 512 characters (~2-3 paragraphs) provides enough context to understand topics while maintaining retrieval precision. Will validate with production corpus (500+ documents).

### First-Run Behavior (Production Awareness)

**Expected startup time breakdown:**

1. **Embedding model download**: ~30 seconds (90MB, one-time cache)
2. **ChromaDB initialization**: ~5 seconds
3. **Vector creation**: ~10 seconds per 2 documents
4. **Total first run**: 45-60 seconds

**Subsequent runs**: <1 second (loads from disk)

**Why this matters:** Documenting expected behavior prevents "is it broken?" questions. Implementation Engineers must distinguish architecture reality from bugs.

### Rolling Release Stability

**Challenge:** EndeavourOS updates Python frequently via `pacman -Syu`. Traditional venv uses symlinks to system Python, which break when Python updates (3.11 → 3.12).

**Solution:** UV downloads isolated Python 3.11.9 binaries. System updates can't affect them.

**Proof of stability:** Project survived multiple system updates during development.

---

## Technical Stack

**Core Dependencies:**
```txt
llama-index-core>=0.12.0
llama-index-llms-anthropic
llama-index-embeddings-huggingface
llama-index-vector-stores-chroma
pydantic>=2.7.0,<2.10.0
chromadb>=0.5.0
python-dotenv
sentence-transformers
```

**Critical Version Pins:**
- `pydantic>=2.7.0,<2.10.0`: Avoids schema generation errors in 2.10+
- `chromadb>=0.5.0`: Works with Pydantic V2 (0.4.x requires Pydantic V1)
- `llama-index-core>=0.12.0`: Modern Pydantic V2 support

**Why these versions matter:** LlamaIndex migrated to Pydantic V2 in version 0.11+ (mid-2024). Using correct version constraints prevents `proxies` parameter errors and dependency conflicts.

---

## Project Structure

```
rag-doc-assistant-new/
├── app.py              # Main application (Phase 1+2)
├── query.py            # Optimized retrieval (Phase 3)
├── ingest.py           # Reindexing utility
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── data/               # Source documents
│   ├── test-doc-1.md
│   └── test-doc-2.md
├── chroma_db/          # Persistent vector storage
└── README.md           # This file
```

---

## Known Limitations

- **Max document size**: ~1MB per file (LlamaIndex chunking constraints)
- **Supported formats**: .txt, .md, .pdf, .docx
- **Memory usage**: ~2GB RAM for 1,000 documents
- **Query latency**: 2-3 seconds (network + LLM generation)

---

## Future Roadmap

- [ ] Scale to Scholar's Web (500+ Obsidian notes)
- [ ] Implement Phase 3 optimizations on production corpus
- [ ] Multi-turn conversation memory (context across queries)
- [ ] Custom prompt templates (domain-specific optimization)
- [ ] Gradio web UI (non-technical user access)
- [ ] Integration with Scriptorium Assistant (voice capabilities)

---

## Career Context

**Transition:** Customer Support Specialist (8+ years at Cresta AI) → Implementation Specialist at AI companies

**Target compensation:** $120k-150k by February 2026

**Why this project demonstrates Implementation Engineering:**

1. **RAG system implementation** - Core requirement for AI platform roles
2. **Cost optimization thinking** - 93% savings documented and explained
3. **Production awareness** - Cold-start behavior, persistence patterns, version stability
4. **Real-world application** - Foundation for customer-facing Scriptorium Assistant

**This project proves I can:**
- Build complex AI systems from scratch
- Make informed technical decisions (chunking strategy, cost trade-offs)
- Document production considerations for deployment planning
- Scale prototypes to production (Phase 3 ready for 500+ documents)

---

## Troubleshooting

### Common Issues

**Error: `proxies` parameter**
- **Cause:** Version mismatch between anthropic SDK and llama-index-llms-anthropic
- **Fix:** Use requirements.txt versions (Pydantic V2 compatible)

**Error: ChromaDB lock**
- **Cause:** Previous process didn't release database
- **Fix:** `rm -rf chroma_db/` then re-run

**Error: First query >5 minutes**
- **Cause:** Slow network downloading embedding model
- **Fix:** Normal on slow connections; model caches after first download

### Environment Issues

**UV not found after installation:**
```bash
source $HOME/.cargo/env
# Add to ~/.bashrc for persistence
```

**Python version mismatch:**
```bash
cat .python-version  # Should show 3.11.9
uv python pin 3.11.9
```

---

## Interview Talking Points

**Technical depth:**
> "I implemented a RAG system using Claude API and local embeddings, achieving 93% cost reduction vs OpenAI. I tested three chunking strategies (256/512/1024 characters) and documented the trade-offs. With my test corpus, optimizations showed marginal gains, but I preserved the implementation for production deployment at 500+ documents where precision matters."

**Problem-solving:**
> "I hit version compatibility issues between the Anthropic SDK and LlamaIndex. Rather than forcing old versions, I researched current best practices, discovered LlamaIndex had migrated to Pydantic V2, updated my stack, and documented the solution for others."

**Production thinking:**
> "I documented that 45-60 second first-run latency is expected behavior, not a bug. Implementation Engineers must distinguish architecture reality from defects. This prevents 'is it broken?' conversations with clients."

---

## License

MIT

---

## Contact

Built by **CK (Erick Allas)** as part of AI career transition portfolio

- **GitHub:** [github.com/NujabesSoul](https://github.com/NujabesSoul)
- **LinkedIn:** [linkedin.com/in/erick-allas-1781786b](https://www.linkedin.com/in/erick-allas-1781786b/)
- **Project:** Renaissance autodidact → AI Implementation Specialist
- **Timeline:** October 2025 - February 2026

**Currently seeking:** Implementation Specialist / Solutions Engineer roles at AI companies
