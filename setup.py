from setuptools import find_packages, setup

setup(
    name="medical-chatbot",
    version="0.1.0",
    author="Medical Chatbot Team",
    author_email="",
    description="A Medical Chatbot using LLMs with RAG (Retrieval-Augmented Generation)",
    packages=find_packages(),
    install_requires=[
        "flask",
        "langchain",
        "langchain-community",
        "langchain-groq",
        "pypdf",
        "faiss-cpu",
        "python-dotenv",
        "sentence-transformers",
        "huggingface-hub",
    ],
    python_requires=">=3.9",
)
