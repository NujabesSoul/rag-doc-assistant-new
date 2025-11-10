"""
RAG Documentation Assistant - Main Application
Built for the Scholar's Brain project

This demonstrates RAG fundamentals: loading documents, creating searchable
vectors, and querying with Claude while citing sources properly.

Phase 1 Goal: Prove end-to-end pipeline works
Phase 2: Add persistence so subsequent runs are fast
Phase 3: Optimize chunking for better retrieval quality
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
import os
import platform

# Load environment variables from .env file
# This pulls in ANTHROPIC_API_KEY without hardcoding it
load_dotenv()

# === SYSTEM CHECK ===
# Always good to know what environment we're running in
print(f"🖥️  Platform: {platform.system()}")
print(f"🐍 Python: {platform.python_version()}")
print("")  # Breathing room in output

# === SECURITY CHECK ===
# Fail fast if API key is missing - better than cryptic error later
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError(
        "❌ ANTHROPIC_API_KEY not found in .env file\n"
        "   Create .env and add: ANTHROPIC_API_KEY=sk-ant-your-key-here"
    )

print("🚀 Starting RAG Documentation Assistant (Scholar's Brain prototype)...")
print("")

# === CONFIGURE THE STRATEGIC CORTEX (Claude) ===
# This is your LLM - the piece that generates natural language answers
print("🤖 Initializing Strategic Cortex (Claude Sonnet 4)...")

strategic_cortex = Anthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Why Sonnet 4? Balance of quality and cost.
# Haiku is cheaper but less nuanced for complex queries.
# Opus is brilliant but expensive for development iteration.
# Sonnet 4 hits the sweet spot for RAG applications.

# === CONFIGURE THE EMBEDDING MODEL (Local) ===
# This converts text into vectors - the "semantic fingerprint" of meaning
print("🔧 Configuring local embedding model (all-MiniLM-L6-v2)...")
print("   📚 This model creates 384-dimensional vectors from text")
print("   💰 Runs locally - no API costs")
print("")

embedding_engine = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Why this model?
# - 384 dimensions is the sweet spot (not too large, not too simple)
# - Trained on semantic similarity tasks (designed for this exact use case)
# - Fast inference (< 100ms per document chunk)
# - Free (no API costs like OpenAI's ada-002)
# Trade-off: Slightly lower quality than ada-002, but 90% as good for 100% less cost

# === SET GLOBAL DEFAULTS ===
# LlamaIndex uses these everywhere unless you override
Settings.llm = strategic_cortex
Settings.embed_model = embedding_engine

# === LOAD DOCUMENTS FROM RESEARCH LIBRARY ===
print("📂 Loading documents from /data (your research library)...")

# SimpleDirectoryReader is smart:
# - Automatically handles .txt, .md, .pdf, .docx
# - Preserves metadata (filename, modification date)
# - Handles encoding issues gracefully
research_library = SimpleDirectoryReader("data").load_data()

document_count = len(research_library)
print(f"✅ Loaded {document_count} documents from your research library")
print("")

# Show what we found (helpful for debugging)
for idx, doc in enumerate(research_library, start=1):
    filename = doc.metadata.get('file_name', 'Unknown')
    char_count = len(doc.text)
    # Format with thousands separator for readability
    print(f"  {idx}. {filename} - {char_count:,} characters")

print("")  # Visual separation before big operation

# === CREATE VECTOR INDEX (The "Scholar's Web" Search Layer) ===
print("🔍 Creating vector index (semantic search layer)...")
print("")
print("   ⏳ FIRST RUN TIMING:")
print("      • Embedding model download: ~30 seconds (one-time, 90MB)")
print("      • ChromaDB initialization: ~5 seconds")
print("      • Vector creation: ~10 seconds (for 2 small docs)")
print("      • Total: ~45-60 seconds")
print("")
print("   ⚡ SUBSEQUENT RUNS (after adding persistence in Phase 2):")
print("      • Load from disk: <1 second")
print("")
print("   🎓 WHAT'S HAPPENING:")
print("      1. Each document is split into chunks (default: ~1000 chars)")
print("      2. Each chunk becomes a 384-dimensional vector")
print("      3. Vectors are stored in ChromaDB for fast similarity search")
print("      4. When you query, your question also becomes a vector")
print("      5. ChromaDB finds chunks with similar vectors (semantic search)")
print("")

# This is where the "magic" happens - but it's just math
# Similar meanings produce similar vectors (cosine similarity)
knowledge_base = VectorStoreIndex.from_documents(research_library)

print("✅ Vector index created successfully")
print("   💡 Your documents are now searchable by meaning, not just keywords")
print("")

# === CREATE QUERY ENGINE ===
# This orchestrates: query → retrieve context → prompt Claude → return answer
query_engine = knowledge_base.as_query_engine()

# Under the hood, query_engine does:
# 1. Convert your question to a vector
# 2. Find top-k most similar document chunks (default k=2)
# 3. Construct prompt: "Here are relevant documents: [context]. Answer this: [question]"
# 4. Send to Claude
# 5. Return response with source citations

# === TEST QUERY (Prove It Works) ===
print("❓ Testing with sample query...")
print("   💰 Cost: ~$0.01 with Claude Sonnet 4")
print("   ⏱️  Response time: ~2-3 seconds (after index creation)")
print("")

test_query = "What are these documents about? Summarize the main topics covered."

# This is your first real RAG query - moment of truth!
answer = query_engine.query(test_query)

print("📄 Claude's Response:")
print("-" * 60)
print(answer)
print("-" * 60)
print("")

# === SHOW SOURCE CITATIONS ===
# This is critical - we can trace WHERE the answer came from
print("📚 Sources Used (with similarity scores):")
print("")

for rank, source_node in enumerate(answer.source_nodes, start=1):
    source_file = source_node.metadata.get('file_name', 'Unknown')
    similarity_score = source_node.score
    
    print(f"  {rank}. {source_file}")
    print(f"     Similarity: {similarity_score:.3f} (higher = more relevant)")
    
    # Show a preview of what was retrieved
    chunk_preview = source_node.text[:150].replace('\n', ' ')
    print(f"     Preview: {chunk_preview}...")
    print("")

# === COMPLETION STATUS ===
print("=" * 60)
print("✅ Phase 1 Complete!")
print("")
print("What you just built:")
print("  • Document loading from /data folder")
print("  • Semantic vector search (meaning-based, not keywords)")
print("  • Claude API integration with proper context")
print("  • Source citation (traceable answers)")
print("")
print("Next Steps:")
print("  • Phase 2: Add persistence (subsequent runs in <1 second)")
print("  • Phase 3: Optimize chunking for better retrieval")
print("  • Phase 4: Build interactive UI (Gradio)")
print("")
print("💡 Cost Breakdown:")
print("   • This query: ~$0.01")
print("   • Embeddings: $0.00 (local model)")
print("   • Total project cost (100 queries): ~$1.00")
print("=" * 60)
