"""System prompts for the Day 1 RAG lab.

The grounding rule is the whole lesson: the model may only use the retrieved
context. In live mode the prompt enforces it; in stub mode the synthesiser
in ``llm.py`` enforces it structurally by picking sentences from the context.
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
