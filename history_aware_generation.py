from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# =========================
# Load environment variables
# =========================
load_dotenv()

# =========================
# Load Chroma DB
# =========================
persist_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# =========================
# Load Local LLM
# =========================
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# distilgpt2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# =========================
# Chat History
# =========================
chat_history = []


# =========================
# Helper: Format history
# =========================
def format_chat_history(history):
    if not history:
        return ""

    formatted = []

    for user_msg, ai_msg in history:
        formatted.append(f"User: {user_msg}")
        formatted.append(f"Assistant: {ai_msg}")

    return "\n".join(formatted)


# =========================
# Helper: Generate Text
# =========================
def generate_text(prompt, max_new_tokens=120):
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
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt from output
    response = generated_text[len(prompt):].strip()

    # Clean weird continuations
    response = response.split("User:")[0]
    response = response.split("Assistant:")[0]

    return response.strip()


# =========================
# Main Question Function
# =========================
def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # =====================================
    # STEP 1: Rewrite query using history
    # =====================================
    if chat_history:

        history_text = format_chat_history(chat_history[-3:])

        rewrite_prompt = f"""
            Conversation History:
            {history_text}

            Current User Question:
            {user_question}

            Rewrite the current question into a standalone search query.

            Search Query:
            """

        rewritten = generate_text(
            rewrite_prompt,
            max_new_tokens=30
        )

        search_question = (
            rewritten
            if rewritten.strip()
            else user_question
        )

        print(f"\nSearching for: {search_question}")

    else:
        search_question = user_question

    # =====================================
    # STEP 2: Retrieve Documents
    # =====================================
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(search_question)

    print(f"\nFound {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        preview = "\n".join(
            doc.page_content.split("\n")[:2]
        )

        print(f"\nDocument {i}:")
        print(preview[:200] + "...")

    # =====================================
    # STEP 3: Build Final Prompt
    # =====================================
    docs_text = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    history_text = format_chat_history(chat_history[-5:])

    final_prompt = f"""
        You are a helpful AI assistant.

        Answer the user's question ONLY using the provided documents.

        If the answer is not found in the documents, say:
        "I don't have enough information to answer that question based on the provided documents."

        Conversation History:
        {history_text}

        Relevant Documents:
        {docs_text}

        User Question:
        {user_question}

        Answer:
        """

    # =====================================
    # STEP 4: Generate Answer
    # =====================================
    answer = generate_text(
        final_prompt,
        max_new_tokens=150
    )

    # Fallback if model outputs garbage/empty
    if not answer.strip():
        answer = (
            "I don't have enough information "
            "to answer that question based on "
            "the provided documents."
        )

    # =====================================
    # STEP 5: Save Chat History
    # =====================================
    chat_history.append(
        (user_question, answer)
    )

    print(f"\nAnswer:\n{answer}")

    return answer


# =========================
# Chat Loop
# =========================
def start_chat():

    print("Ask me questions!")
    print("Type 'quit' to exit.")

    while True:

        question = input("\nYour Question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


# =========================
# Run
# =========================
if __name__ == "__main__":
    start_chat()