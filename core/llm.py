from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)


def generate_response(user_query: str, retrieved_context: str) -> str:
    prompt = f"""You are a helpful college placements assistant for VNR Vignana Jyothi Institute of Engineering and Technology.

Answer the user's question using ONLY the provided context below.
- Use company names exactly as they appear in the context — never use internal IDs.
- If the answer is not in the context, reply exactly: "I don't have that information in my current placements database."
- For greetings (Hi, Hello, etc.), respond warmly and explain how you can help with placement queries.

Context:
{retrieved_context}

Question: {user_query}

Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.GROQ_SMART_MODEL,
        temperature=0.0,
    )
    return response.choices[0].message.content
