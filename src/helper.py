from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_data(data_dir: str = "data/"):
    """Load all PDF files from a directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def split_text(extracted_data, chunk_size: int = 500, chunk_overlap: int = 20):
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_huggingface_embeddings():
    """Download and return HuggingFace sentence-transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
