from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


# Chunking type 1: ChatracterTextSplitter
char_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=100, 
    chunk_overlap=0
)

chunks1 = char_splitter.split_text(tesla_text)
print("=" * 50)
print("CharacterTextSplitter Chunks:")
print("=" * 50)
for i, chunk in enumerate(chunks1, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()


# Chunking type 2: RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100, 
    chunk_overlap=0
)

chunks2 = recursive_splitter.split_text(tesla_text)
print("=" * 50)
print("RecursiveCharacterTextSplitter Chunks:")
print("=" * 50)
for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()


# Chunking type 3: SemanticChunker
semantic_splitter = SemanticChunker(
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=70
)

chunks3 = semantic_splitter.split_text(tesla_text)
print("=" * 50)
print("SemanticChunker Chunks:")
print("=" * 50)
for i, chunk in enumerate(chunks3, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()


# Chunking type 4: AI Agentic Chunking
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# TinyLlama needs a padding token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

prompt = f"""
You are a text chunking expert.

Split the following text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put <<<SPLIT>>> between chunks
- Return ONLY the chunked text

Text:
{tesla_text}

Chunked Text:
"""

print("🤖 Asking AI to chunk the text...")

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024
).to(device)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# Remove prompt from generated text
marked_text = generated_text[len(prompt):].strip()

chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []

for chunk in chunks:

    cleaned = chunk.strip()

    if cleaned:
        clean_chunks.append(cleaned)

print("=" * 50)
print("AI Agentic Chunker:")
print("=" * 50)
for i, chunk in enumerate(clean_chunks, 1):

    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
