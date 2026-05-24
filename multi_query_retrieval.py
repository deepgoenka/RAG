from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


persistent_directory = "db/chroma_db"

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)



model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

model.eval()


def generate_response(
    prompt,
    max_new_tokens=150
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    response = generated_text[
        len(prompt):
    ].strip()

    return response


def generate_query_variations(
    original_query,
    num_queries=3
):

    prompt = f"""
        You are a search query rewriting assistant.

        Generate {num_queries} different search queries
        that mean the same thing as the original query.

        Make each variation:
        - semantically similar
        - differently phrased
        - useful for document retrieval

        Original Query:
        {original_query}

        Return ONLY the queries.
        One query per line.

        Queries:
        """

    response = generate_response(
        prompt,
        max_new_tokens=100
    )


    lines = response.split("\n")

    cleaned_queries = []

    for line in lines:

        line = line.strip()

        # Remove numbering
        line = re.sub(
            r"^\d+[\).\s-]*",
            "",
            line
        )

        if (
            line
            and len(line) > 5
            and line not in cleaned_queries
        ):
            cleaned_queries.append(line)

    # Ensure original query included
    if original_query not in cleaned_queries:
        cleaned_queries.insert(0, original_query)

    return cleaned_queries[:num_queries]


original_query = "How does Tesla make money?"

print(f"\nOriginal Query:")
print(original_query)

print("\n" + "=" * 60)

# Generate Variations
query_variations = generate_query_variations(
    original_query,
    num_queries=3
)

print("\nGenerated Query Variations:\n")

for i, variation in enumerate(
    query_variations,
    1
):

    print(f"{i}. {variation}")

print("\n" + "=" * 60)


# Retrieval
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

all_retrieval_results = []

for i, query in enumerate(
    query_variations,
    1
):

    print(
        f"\n=== RESULTS FOR QUERY {i} ==="
    )

    print(f"Query: {query}\n")

    docs = retriever.invoke(query)

    all_retrieval_results.append(docs)

    print(
        f"Retrieved {len(docs)} documents:\n"
    )

    for j, doc in enumerate(
        docs,
        1
    ):

        print(f"Document {j}:\n")

        preview = (
            doc.page_content[:150]
            .replace("\n", " ")
        )

        print(f"{preview}...\n")

    print("-" * 50)


print("\n" + "=" * 60)
print("✅ Multi-Query Retrieval Complete!")
