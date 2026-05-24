import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader    # Help reading text files from a directory
from langchain_text_splitters import CharacterTextSplitter                       # For chunking
from langchain_community.embeddings import HuggingFaceEmbeddings                 # Free local embeddings (sentence-transformers)
from langchain_community.vectorstores import Chroma                             # Vector database
from dotenv import load_dotenv                                                  # Loading environment variables from a .env file

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    # Check if the docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist. Please create it and add your .txt files.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        docs_path, 
        glob="*.txt",           # Only look for txt files
        loader_cls=TextLoader   # Using TextLoader for reading text files
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in '{docs_path}'. Please add some text files to ingest.")
    
    for i, doc in enumerate(documents[:2]):     # Show first 2 docs
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content Length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print(f"\nSplitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):     # Show first 5 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Length: {len(chunk.page_content)} characters")
            print(f"  Content: {chunk.page_content}")
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"\nCreating embeddings and storing in Chroma vector database")

    # Initialize a free local embedding model (downloaded on first run)
    # You can switch to: "BAAI/bge-small-en-v1.5" for often-better retrieval quality.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Create the Chroma vector store
    print(f"Creating Chroma vector store")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---Finished creating vector store---")

    print("Vector store created and saved to {persist_directory}")

    return vector_store


def main():
    print("Starting the ingestion pipeline...")

    # Step 1: Loading the files
    documents = load_documents(docs_path="docs")

    # Step 2: Chunking the files
    chunks = split_documents(documents)

    # Step 3: Embedding and Storing in Vector DB
    vector_store = create_vector_store(chunks)



if __name__ == "__main__":
    main()