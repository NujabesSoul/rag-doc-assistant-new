"""
Chunking Experiment: Compare 256/512/1024 chunk sizes
Educational tool to understand chunking trade-offs
"""

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Load test documents
docs = SimpleDirectoryReader("../data").load_data()

print("📊 CHUNKING COMPARISON")
print("=" * 60)
print("")

# Test different chunk sizes
for chunk_size in [256, 512, 1024]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    
    chunks = splitter.get_nodes_from_documents(docs)
    
    total_chars = sum(len(chunk.text) for chunk in chunks)
    avg_size = total_chars / len(chunks)
    
    print(f"Chunk Size: {chunk_size}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk size: {avg_size:.0f} characters")
    print(f"  Coverage: {total_chars:,} characters total")
    print("")
    
    # Show sample chunk
    if len(chunks) > 0:
        sample = chunks[0].text[:200].replace('\n', ' ')
        print(f"  Sample chunk: {sample}...")
        print("")
    
    print("-" * 60)
    print("")

print("🎓 INSIGHTS:")
print("  • 256: High precision, but many chunks (slower retrieval)")
print("  • 512: Sweet spot for Q&A (balance of precision/context)")
print("  • 1024: Fewer chunks, but less precise (can retrieve irrelevant context)")
