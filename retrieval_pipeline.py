from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

persist_directory = "db/chroma_db"  # Directory where the embeddings are stored

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Recreating the vector embeddings
db = Chroma(
    collection_metadata={"hnsw:space": "cosine"},  # while dealing with RAG, generally cosine similarity is used
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

query = "What was NVIDIA's first graphics accelerator called?"  # Example query to test the retrieval

retriever = db.as_retriever(search_kwargs={"k": 3})  # Top 3 chunks with the highest similarity score will be retrieved

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("---Context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# combine query and relevant documents contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f'- {doc.page_content}' for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents.
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Free local LLM via HuggingFace Transformers (no OpenAI, no Ollama required)
# Downloads model on first run.
# distilgpt2 is small and widely available, but not great for strict QA.
# Swap model_name to a better small chat/QA model if you want.
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# distilgpt2 doesn't support real chat; treat combined_input as the prompt.
# Add a stricter instruction to reduce looping/multiple continuations.
prompt = combined_input + "\n\nAnswer (one line only):"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
completion = generated[len(prompt) :].strip()

# Keep only the first line to prevent repeated "Answer:" continuations.
completion = completion.splitlines()[0].strip() if completion else ""

print("\n---Generated Response---")
print("Content only: ")
print(completion)

