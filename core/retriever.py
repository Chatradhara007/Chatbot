def get_retriever(vector_db, top_k=5):
    return vector_db.as_retriever(search_kwargs={"k": top_k})
