from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)


def route_query(user_query: str) -> str:
    """
    Route to ANALYTICAL (pandas/MongoDB computation) or SEMANTIC (vector DB).
    """
    prompt = f"""You are a routing assistant for a college placements chatbot.

Classify the user query into exactly one of these two routes:

ANALYTICAL — Use this when the query needs aggregation, math, or ranking:
- Highest / lowest / best / worst package, salary, CTC, LPA
- "Which company offered the highest package?" → ANALYTICAL
- Average salary, total offers, total students placed
- Comparing multiple companies by a numeric metric
- Top N companies by package / offers
- Placement percentage, selection rate, competition ratio
- Branch-wise statistics (placed, registered, average salary)
- Batch-wise statistics (total students)
- Internship vs full-time count or salary breakdown
- "How many offers did X give?"
- Any question requiring finding a winner/loser/rank by a number

SEMANTIC — Use this when the query asks for descriptive details:
- "Tell me about Amazon's drive"
- "What roles does TCS offer?"
- "Which branches are eligible for Google?"
- "What is the company description for Infosys?"
- "What sector does Wipro belong to?"
- Job locations, drive type (on-campus/off-campus)
- Skills required by a specific company
- Any question NOT involving math, rankings, or aggregation
- Greetings and general conversation

Query: "{user_query}"
Route:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.GROQ_SMART_MODEL,
        temperature=0.0,
    )
    route = response.choices[0].message.content.strip().upper()
    return "ANALYTICAL" if "ANALYTICAL" in route else "SEMANTIC"
