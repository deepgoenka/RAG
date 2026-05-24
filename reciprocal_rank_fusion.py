# Reciprocal Rank Fusion Score = Σ (1/(k + rank_position))
# where k is a constant (often set to 60) to dampen the effect of lower-ranked results
#       rank_position is the position of the document in the ranked list
#       the sum is across all queries where the chunk appears

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from collections import defaultdict

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



# RRF Implementation
def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)


print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")
