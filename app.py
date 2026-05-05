import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from src.helper import download_huggingface_embeddings
from src.prompt import system_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------------------------------------------------------
# Initialise embeddings & local FAISS vector store
# ---------------------------------------------------------------------------
embeddings = download_huggingface_embeddings()

FAISS_INDEX_PATH = "faiss_index"

# Load existing FAISS index from disk
docsearch = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ---------------------------------------------------------------------------
# Build LangChain retrieval chain with Groq
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"answer": "Please enter a valid question."})

    try:
        response = rag_chain.invoke({"input": user_message})
        answer = response.get("answer", "Sorry, I could not find an answer.")
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    return jsonify({"answer": answer})


if __name__ == "__main__":
    print("--- Medical AI Chatbot ---")
    print("App is starting on http://localhost:8080")
    print("Please use PORT 8080 in your browser.")
    app.run(host="0.0.0.0", port=8080, debug=True)
