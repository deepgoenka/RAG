# Retrieval-Augmented Generation (RAG)

References:
1. [https://www.youtube.com/embed/videoseries?si=B4GLBCoMLecxRdV1&amp;list=PLNIQLFWpQMRUMjxfe8o6g3uzJ6LH_VotY](https://youtube.com/playlist?list=PLNIQLFWpQMRUMjxfe8o6g3uzJ6LH_VotY&si=eQOS3pa0olS8sDAR)
2. Vector Database: https://github.com/deepgoenka/RAG/blob/main/Vector_Database.md
3. ChromaDB:
     * Documentation: https://docs.trychroma.com/docs/overview/introduction
     * YouTube Video: https://youtu.be/Qs_y0lTJAp0?si=QBG3890snbPioFAA
5. Pinecone:
     * Documentation: https://docs.pinecone.io/guides/get-started/overview
     * YouTube Video: https://youtu.be/3OGWzDNMeaQ?si=hhG2E_11dXqerML5
7. FAISS:
     * Documentation: https://faiss.ai/index.html
     * YouTube Video: https://youtu.be/sKyvsdEv6rk?si=1PHgJyj7eOag_pNt
9. Cohere Reranker: https://cohere.com/rerank


<img width="1586" height="1200" alt="RAG Pipeline" src="https://github.com/user-attachments/assets/c7e536a7-a536-4144-a67f-a838b80981ba" />


## ChromaDB

Chroma is an open-source vector database mainly designed for AI applications and Retrieval-Augmented Generation (RAG). It is beginner-friendly and integrates easily with frameworks like LangChain and LlamaIndex.

### Key Points

* Open-source and lightweight
* Easy local setup
* Stores embeddings + metadata + documents
* Good for prototyping and small-to-medium AI apps
* Works well with Python

### Best Use Cases

* RAG applications
* AI chatbots
* Semantic search
* Personal AI projects

---

## Pinecone

Pinecone is a managed cloud-based vector database service optimized for large-scale production AI systems. It handles infrastructure, scaling, indexing, and performance automatically.

### Key Points

* Fully managed cloud service
* Highly scalable
* Fast similarity search
* Production-ready infrastructure
* No need to manage servers

### Best Use Cases

* Enterprise AI systems
* Large-scale semantic search
* Recommendation systems
* Production RAG pipelines

---

## FAISS

FAISS is a high-performance library developed by Meta for efficient similarity search and clustering of dense vectors.

### Key Points

* Extremely fast vector similarity search
* Optimized for CPU and GPU
* Library, not a full database
* Requires manual storage management
* Widely used in research and high-performance systems

### Best Use Cases

* Research projects
* High-speed vector search
* Large embedding datasets
* Custom vector search systems

---

# Difference Between ChromaDB, Pinecone, and FAISS

| Feature                   | ChromaDB              | Pinecone                | FAISS                   |
| ------------------------- | --------------------- | ----------------------- | ----------------------- |
| Type                      | Open-source Vector DB | Managed Cloud Vector DB | Vector Search Library   |
| Setup                     | Easy local setup      | Cloud-based             | Manual setup            |
| Scalability               | Medium                | Very High               | High                    |
| Beginner Friendly         | Yes                   | Moderate                | Less                    |
| Cloud Support             | Optional              | Native                  | Manual                  |
| Metadata Storage          | Yes                   | Yes                     | Limited                 |
| Persistence               | Yes                   | Yes                     | Manual                  |
| GPU Support               | Limited               | Managed internally      | Excellent               |
| Best For                  | Learning & RAG        | Production AI apps      | High-performance search |
| Infrastructure Management | Self-managed          | Fully managed           | Fully manual            |

---

* **ChromaDB** → “SQLite for Vector Databases”
* **Pinecone** → “AWS-managed Vector Database”
* **FAISS** → “Fast Vector Search Engine Library”
