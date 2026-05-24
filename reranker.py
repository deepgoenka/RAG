from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from dotenv import load_dotenv

load_dotenv()

chunks = [
    # Tesla - Financial & Production
    "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
    "Tesla's automotive gross margin improved to 19.3% this quarter.",
    "Tesla Cybertruck production ramp begins in 2024 with initial deliveries.",
    "Tesla announced plans to expand Gigafactory production capacity.",
    "Tesla stock price reached new highs following earnings announcement.",
    "Tesla's energy storage business grew 40% year-over-year.",
    
    # Microsoft - Development & Acquisitions
    "Microsoft acquired GitHub for $7.5 billion in 2018.",
    "Microsoft's cloud revenue Azure grew 29% year-over-year.",
    "Microsoft announced new AI features for Visual Studio Code.",
    "Microsoft Teams integration with GitHub enhances developer workflow.",
    "Microsoft's developer tools division sees strong adoption.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",
    
    # NVIDIA - AI & Hardware
    "NVIDIA's data center revenue reached $47.5 billion annually.",
    "NVIDIA's H100 GPUs see unprecedented demand for AI training.",
    "NVIDIA announced next-generation Blackwell architecture.",
    "NVIDIA's gaming revenue declined due to crypto market changes.",
    "NVIDIA's automotive AI platform partnerships expanded.",
    "NVIDIA's AI chip shortage affects cloud providers.",
    
    # Google/Alphabet - AI & Cloud
    "Google's AI investments total over $100 billion in recent years.",
    "Google Cloud revenue grew 35% reaching $8.4 billion quarterly.",
    "Google announced Gemini AI model competing with GPT-4.",
    "Google's search advertising revenue remains strong at $59 billion.",
    "Google's Workspace products integrate advanced AI features.",
    "Google announced quantum computing breakthroughs.",
    "Google's autonomous vehicle division Waymo expands operations.",
    
    # Noisy/Less Relevant Chunks
    "The Tesla coil was invented by Nikola Tesla in 1891.",
    "Microsoft Excel spreadsheet formulas can be complex for beginners.",
    "NVIDIA Shield TV streaming device gets software update.",
    "Google Maps navigation improved with real-time traffic data.",
    "Production delays affected multiple manufacturing sectors.",
    "Financial markets showed volatility during earnings season.",
    "Revenue recognition standards changed for software companies.",
    "Hardware components face supply chain constraints globally."
]

print(f"Created {len(chunks)} sample chunks for demonstration")


# Convert to Document objects for LangChain
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)


# Vector Retrieval / Dense Retrieval / Semantic Search
print("Setting up Vector Retriever...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})


# BM25 Retriever / Keyword Search / Sparse Retrieval
print("Setting up BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15


# Hybrid Retriever - EnsembleRetriever
print("Setting up Hybrid Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Equal weight to vector and keyword search
)


query = "Tesla financial performance and production updates"

print("STEP 1: Hybrid Search Results")
print("-"*50)

retrieved_docs = hybrid_retriever.invoke(query)  # Get top 25 for reranking

# Show top 10 from hybrid search
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print(f"\n(Retrieved {len(retrieved_docs)} total chunks for reranking)\n")



print("Applying Cohere Reranking")
print("-"*50)

# Initialize Cohere reranker
reranker = CohereRerank(model="rerank-english-v3.0", top_n=10)

# Rerank the retrieved documents
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# Show reranked results
for i, doc in enumerate(reranked_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print("\n" + "="*80)
print("ANALYSIS:")
print("✅ Hybrid Search: Mixed relevant and irrelevant results")
print("✅ Reranking: Most relevant Tesla financial/production info at top")
print("✅ Notice how reranking moved the most contextually relevant chunks higher")