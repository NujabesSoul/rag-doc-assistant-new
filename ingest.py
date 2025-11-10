"""
Document Reindexing Utility
Use this when you've added/removed/modified documents in /data

This script:
1. Deletes old vector index
2. Reloads all documents from /data
3. Creates fresh vectors
4. Saves to ChromaDB

Run with: uv run python ingest.py
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import chromadb
import shutil
import os
from datetime import datetime

load_dotenv()

print("=" * 70)
print("🔄 RAG DOCUMENTATION ASSISTANT - REINDEXING UTILITY")
print("=" * 70)
print("")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("")

# === SAFETY CHECK ===
# Confirm before deleting existing index
if os.path.exists("./chroma_db"):
    vector_count = "unknown"
    try:
        temp_db = chromadb.PersistentClient(path="./chroma_db")
        temp_collection = temp_db.get_or_create_collection("documents")
        vector_count = len(temp_collection.get()['ids'])
    except:
        pass
    
    print(f"⚠️  Found existing index: {vector_count} vectors")
    print("   This will be permanently deleted and rebuilt from /data documents")
    print("")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n❌ Reindexing cancelled. Your existing index is unchanged.")
        exit(0)
    
    print("\n🗑️  Removing old index...")
    shutil.rmtree("./chroma_db")
    print("✅ Old index deleted")
    print("")

# === CONFIGURE EMBEDDING ENGINE ===
print("🔧 Configuring embedding engine...")
embedding_engine = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.embed_model = embedding_engine
print("✅ Embedding engine ready")
print("")

# === CREATE FRESH CHROMADB ===
print("🗄️  Creating fresh ChromaDB...")
vector_db = chromadb.PersistentClient(path="./chroma_db")
doc_collection = vector_db.get_or_create_collection("documents")

vector_store = ChromaVectorStore(chroma_collection=doc_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("✅ Fresh database created")
print("")

# === LOAD DOCUMENTS ===
print("📂 Loading documents from /data...")

try:
    research_library = SimpleDirectoryReader("data").load_data()
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    print("   Check that /data folder exists and contains readable files")
    exit(1)

doc_count = len(research_library)

if doc_count == 0:
    print("⚠️  No documents found in /data folder!")
    print("   Add .txt, .md, or .pdf files to /data and try again")
    exit(1)

print(f"✅ Loaded {doc_count} documents")
print("")

# === SHOW WHAT'S BEING INDEXED ===
print("📋 Documents to be indexed:")
print("")

total_chars = 0
for idx, doc in enumerate(research_library, start=1):
    filename = doc.metadata.get('file_name', 'Unknown')
    char_count = len(doc.text)
    total_chars += char_count
    
    print(f"  {idx}. {filename}")
    print(f"     Size: {char_count:,} characters")

print("")
print(f"Total content: {total_chars:,} characters")
print("")

# === CREATE VECTOR INDEX ===
print("🔍 Creating vector index...")
print("   ⏳ This will take ~45-60 seconds")
print("   🎓 Each document is: chunked → embedded → stored")
print("")

try:
    knowledge_base = VectorStoreIndex.from_documents(
        research_library,
        storage_context=storage_context,
        show_progress=True
    )
except Exception as e:
    print(f"\n❌ Error creating index: {e}")
    exit(1)

# === VERIFICATION ===
final_vector_count = len(doc_collection.get()['ids'])

print("")
print("=" * 70)
print("✅ REINDEXING COMPLETE!")
print("")
print(f"Documents indexed: {doc_count}")
print(f"Vectors created: {final_vector_count}")
print(f"Storage location: ./chroma_db/")
print(f"Estimated disk usage: ~{final_vector_count * 4}KB")
print("")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print("")
print("Next steps:")
print("  • Run 'uv run python app.py' to start querying")
print("  • Your index is now up-to-date with /data contents")
