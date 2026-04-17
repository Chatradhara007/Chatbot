from langchain_community.vectorstores import FAISS
from core.embedder import get_embedder


def create_and_save_index(documents, save_path):
    embedder = get_embedder()
    print("Generating embeddings and building FAISS index...")
    vector_db = FAISS.from_documents(documents, embedder)
    vector_db.save_local(save_path)
    print("Vector database saved successfully.")


def load_index(save_path):
    embedder = get_embedder()
    return FAISS.load_local(save_path, embedder, allow_dangerous_deserialization=True)
