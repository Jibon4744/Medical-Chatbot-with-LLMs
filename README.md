# 🩺 Medical Chatbot with LLMs

A **Retrieval-Augmented Generation (RAG)** medical chatbot powered by **LangChain**, **FAISS** (local vector store), **HuggingFace Embeddings**, and **Groq** (Mixtral-8x7B), served through a polished **Flask** web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![FAISS](https://img.shields.io/badge/FAISS-Local_Vector_DB-blue)

---

## 📋 How It Works

1. **Ingest** — Medical PDFs are loaded, split into chunks, and embedded using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Index** — Embeddings are stored locally in a **FAISS** index (no cloud required).
3. **Retrieve** — When a user asks a question, the most relevant chunks are retrieved via similarity search.
4. **Generate** — The retrieved context + question are sent to **Groq (Mixtral-8x7B)**, which generates a grounded answer.

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/Medical-Chatbot-with-LLMs.git
cd Medical-Chatbot-with-LLMs

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Environment Variables

Edit the `.env` file with your Groq API key (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Add Medical Data

Place your medical PDF files in a `data/` directory:

```
data/
  ├── medical_book_1.pdf
  └── medical_book_2.pdf
```

### 4. Build the Vector Index

```bash
python store_index.py
```

This will load the PDFs, chunk them, compute embeddings, and save a local FAISS index to `faiss_index/`.

### 5. Run the Application

```bash
python app.py
```

Open **http://localhost:8080** in your browser to start chatting!

---

## 🗂️ Project Structure

```
Medical-Chatbot-with-LLMs/
├── app.py              # Flask web application
├── store_index.py      # One-time index builder script
├── setup.py            # Package configuration
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── faiss_index/        # Local FAISS vector index (auto-generated)
├── src/
│   ├── __init__.py     # Package exports
│   ├── helper.py       # PDF loading, text splitting, embeddings
│   └── prompt.py       # System prompt template
├── templates/
│   └── chat.html       # Chat UI (dark theme)
├── data/               # Place medical PDFs here
└── research/
    └── trials.ipynb    # Experimentation notebook
```

---

## ⚙️ Tech Stack

| Component     | Technology                              |
| ------------- | --------------------------------------- |
| LLM           | Groq — Mixtral-8x7B-32768 (free API)  |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2`         |
| Vector Store  | FAISS (local, no cloud needed)          |
| Framework     | LangChain                              |
| Web Server    | Flask                                  |
| Frontend      | Vanilla HTML/CSS/JS (dark glassmorphic) |

---

## ⚠️ Disclaimer

This chatbot is for **informational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 📄 License

This project is licensed under the Apache-2.0 License — see the [LICENSE](LICENSE) file for details.