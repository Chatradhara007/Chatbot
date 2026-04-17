from langchain_huggingface import HuggingFaceEmbeddings
import config


def get_embedder():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)