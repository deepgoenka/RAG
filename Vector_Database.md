# What is a Vector Database?

A vector database is a specialized type of database designed to store, index and search high dimensional vector representations of data known as embeddings. Unlike traditional databases that rely on exact matches vector databases use similarity search techniques such as cosine similarity or Euclidean distance to find items that are semantically or visually similar.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/6e454371-2b2f-4fdd-856a-d4488ebf2870" />

# Embeddings

* Embeddings are dense numerical representations of data such as words, sentences, images or audio mapped into a continuous high dimensional space where similar items are positioned closer together.
* Machine learning models that capture semantic meaning, context and relationships within the data generates them.
* Instead of comparing raw text or media directly embeddings allow systems to measure similarity through mathematical distance metrics like cosine similarity or Euclidean distance for faster search and extraction.
* This makes them important for tasks such as semantic search, recommendation systems, clustering, classification and cross lingual matching.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/76f70d94-522c-457c-8975-7bb08483ba6c" />

# How do they Work?

* Embeddings work by converting raw data like text, images or audio into dense numerical vectors that preserve meaning and relationships.
* First the input is processed through a model such as a transformer for text or a CNN for images to extract key features.
* These features are then encoded into fixed length vectors in a high dimensional space where similar items are positioned close together and dissimilar ones are farther apart.
* This spatial arrangement allows similarity to be measured mathematically enabling applications like search, recommendations and classification to operate based on meaning rather than exact matches.

# Popular Vector Databases

* **Pinecone**: Fully managed, cloud native vector database with high scalability and low latency search.
* **Weaviate**: Open source, supports hybrid (keyword + vector) search and offers built in machine learning modules.
* **Milvus**: Highly scalable, open source database optimized for large scale similarity search.
* **Qdrant**: Open source, focuses on high recall, performance and ease of integration with AI applications.
* **Chromadb**: Lightweight, developer friendly vector database often used in LLM powered applications.
