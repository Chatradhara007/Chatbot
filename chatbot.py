import os
from core.loader import load_and_chunk_data
from core.vector_db import create_and_save_index, load_index
from core.retriever import get_retriever
from core.llm import generate_response
from core.analyzer import get_data_agent
from core.router import route_query
import config

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "heya", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "sup", "yo"
}

GREETING_RESPONSE = (
    "Hello! 👋 I'm the VNR Placements Assistant.\n\n"
    "I can help you with:\n"
    "  • Company-wise placement details (roles, packages, offers)\n"
    "  • Branch-wise placement statistics\n"
    "  • Highest & average salary data\n"
    "  • Analytical queries like 'which company gave the most offers?'\n\n"
    "Ask me anything about VNR placements!"
)


def is_greeting(query: str) -> bool:
    return query.lower().strip().rstrip("!.,?") in GREETINGS


def setup_system(force_rebuild: bool = False):
    if force_rebuild or not os.path.exists(config.FAISS_INDEX_PATH):
        reason = "Forced rebuild" if force_rebuild else "First-time setup"
        print(f"{reason}: Building Vector DB (companies + branches + batches)...")
        docs = load_and_chunk_data()
        create_and_save_index(docs, config.FAISS_INDEX_PATH)
        print(f"[Setup] Indexed {len(docs)} documents into FAISS.")
    return load_index(config.FAISS_INDEX_PATH)


def main():
    import sys
    force_rebuild = "--rebuild" in sys.argv

    print("Loading text database...")
    vector_db = setup_system(force_rebuild=force_rebuild)
    retriever = get_retriever(vector_db)

    print("Loading analytics engine...")
    analytical_agent = get_data_agent()

    print("\n" + "=" * 40)
    print("🎓 VNR Placements Chatbot Online!")
    print("Type 'exit' to quit.")
    print("=" * 40 + "\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ['exit', 'quit']:
            print("Shutting down. Goodbye!")
            break

        # ✅ Handle greetings directly — no LLM call needed
        if is_greeting(query):
            print(f"\nBot: {GREETING_RESPONSE}\n")
            continue

        route = route_query(query)
        print(f"[Debug] Route: {route}")

        if route == "ANALYTICAL":
            try:
                answer = analytical_agent(query)
            except Exception as e:
                print(f"[Debug] Agent error: {e}")
                answer = "I had trouble calculating that. Could you rephrase your question?"
        else:
            relevant_docs = retriever.invoke(query)
            context_string = "\n\n".join([doc.page_content for doc in relevant_docs])
            answer = generate_response(query, context_string)

        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()