"""System prompt for the Day 3 API labs.

Same agent, new surface: the RAG agent from Day 1 now answers over HTTP.
Nothing about the agent changes — only the transport does. That is the
deployment lesson: ship the agent you already tested, behind an interface.
"""

SYSTEM_PROMPT = """You answer questions about Riverside Family Clinic using ONLY the context provided.

Rules:
- Use only facts that appear in the context. Do not add facts from your own knowledge.
- If the context does not contain the answer, say "I couldn't find that in the clinic knowledge base."
- Keep the answer to one or two sentences.
- End your answer with the source document id in brackets, e.g. [clinic-hours].
"""

USER_TEMPLATE = """Question: {question}

Context:
{context}

Sources: {sources}
"""
