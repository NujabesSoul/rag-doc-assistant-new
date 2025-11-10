"""
RAG Documentation Assistant - Main Application
Phase 2: Now with persistent storage!

What changed: ChromaDB now saves vectors to disk. Second run loads 
in <1 second instead of rebuilding for 45-60 seconds.
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

print("🚀 Starting RAG Documentation Assistant (Phase 2: Persistent Storage)...")
print("")

# === CONFIGURE STRATEGIC CORTEX (Claude) ===
strategic_cortex = Anthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# === CONFIGURE EMBEDDING ENGINE ===
embedding_engine = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set as global defaults
Settings.llm = strategic_cortex
Settings.embed_model = embedding_engine

# === INITIALIZE PERSISTENT CHROMADB ===
print("🗄️  Initializing ChromaDB with persistent storage...")
print("   📁 Storage location: ./chroma_db/")
print("")

# PersistentClient saves everything to ./chroma_db folder
# This folder will contain SQLite database + vector index files
vector_db = chromadb.PersistentClient(path="./chroma_db")

# Collections are like tables in a database
# We'll use one collection called "documents" for all our content
doc_collection = vector_db.get_or_create_collection("documents")

# Wrap ChromaDB in LlamaIndex's vector store interface
vector_store = ChromaVectorStore(chroma_collection=doc_collection)

# StorageContext tells LlamaIndex where to save things
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === CHECK IF INDEX ALREADY EXISTS ===
# Smart loading: use existing vectors if available, create if not

existing_vector_count = len(doc_collection.get()['ids'])

if existing_vector_count > 0:
    # FAST PATH: Load from disk
    print(f"✅ Found existing index with {existing_vector_count} vectors")
    print("   ⚡ Loading from disk (fast path)...")
    print("")
    
    knowledge_base = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    print("✅ Index loaded in <1 second!")
    print("   💡 This is why persistence matters")
    print("")
    
else:
    # SLOW PATH: Create from scratch (first run only)
    print("📂 No existing index found. Creating from documents...")
    print("   ⏳ This will take 45-60 seconds (one-time initialization)")
    print("")
    
    # Load documents
    research_library = SimpleDirectoryReader("data").load_data()
    doc_count = len(research_library)
    print(f"✅ Loaded {doc_count} documents")
    
    # Show what we're indexing
    for idx, doc in enumerate(research_library, start=1):
        filename = doc.metadata.get('file_name', 'Unknown')
        char_count = len(doc.text)
        print(f"  {idx}. {filename} - {char_count:,} characters")
    
    print("")
    print("🔍 Creating vector index...")
    print("   🎓 Process: Document → Chunks → Embeddings → ChromaDB → Disk")
    print("")
    
    # This creates vectors AND saves them to disk
    knowledge_base = VectorStoreIndex.from_documents(
        research_library,
        storage_context=storage_context,
        show_progress=True  # Shows progress bar
    )
    
    final_vector_count = len(doc_collection.get()['ids'])
    print("")
    print(f"✅ Index created and saved to ChromaDB")
    print(f"   📊 Total vectors: {final_vector_count}")
    print(f"   💾 Disk usage: ~{final_vector_count * 4}KB (4KB per vector average)")
    print("")

# === CREATE QUERY ENGINE ===
query_engine = knowledge_base.as_query_engine()

# === INTERACTIVE QUERY LOOP ===
# Now you can ask multiple questions without rebuilding the index

print("=" * 70)
print("🎯 RAG SYSTEM READY - Interactive Query Mode")
print("")
print("Ask questions about your documents. Type 'exit', 'quit', or 'q' to stop.")
print("")
print("💡 Tips:")
print("   • Ask specific questions for better answers")
print("   • Reference concepts from your documents")
print("   • Each query costs ~$0.01 with Claude Sonnet 4")
print("")
print("=" * 70)

while True:
    # Get user input
    user_question = input("\n❓ Your question: ")
    
    # Exit conditions
    if user_question.lower().strip() in ['exit', 'quit', 'q', '']:
        print("\n👋 Ending session. Your index is saved for next time!")
        print(f"   📁 Vector database: ./chroma_db/ ({existing_vector_count} vectors)")
        break
    
    # Skip empty questions
    if not user_question.strip():
        continue
    
    # Query the system
    print("\n🤔 Searching your research library...")
    
    try:
        answer = query_engine.query(user_question)
        
        # Show answer
        print("\n📄 Answer:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        # Show sources
        print("\n📚 Sources:")
        for rank, source_node in enumerate(answer.source_nodes, start=1):
            source_file = source_node.metadata.get('file_name', 'Unknown')
            similarity = source_node.score
            
            print(f"\n  {rank}. {source_file} (similarity: {similarity:.3f})")
            
            # Show relevant chunk preview
            chunk_preview = source_node.text[:200].replace('\n', ' ')
            print(f"     Context: {chunk_preview}...")
    
    except Exception as e:
        print(f"\n❌ Error processing query: {e}")
        print("   Try rephrasing your question or check your API key.")
