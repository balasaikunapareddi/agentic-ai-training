"""Keyword retrieval over the shared clinic knowledge base.

Deliberately simple — token-overlap ranking, no embeddings, no vector DB —
so the retrieval step is inspectable in a training lab. The same documents
feed the Day 1 RAG lab and the Day 2 research labs; the pattern (retrieve,
then synthesise only from what was retrieved) is the one used in the
``agent-graph-rag`` portfolio repo, minus the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can",
    "could", "how", "what", "when", "where", "why", "i", "you", "we", "it",
    "of", "to", "in", "on", "for", "and", "or", "at", "by", "with", "my",
}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def chunks(self) -> list[str]:
        """Split on blank lines: one chunk per paragraph (heading excluded)."""
        text = self.text
        if text.startswith("#"):
            text = text.split("\n", 1)[-1]
        return [c.strip() for c in text.split("\n\n") if c.strip()]


def _stem(token: str) -> str:
    """Naive stemming so 'book' matches 'booked'/'booking' and 'appointment'
    matches 'appointments'. Applied identically to queries and chunks."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def load_documents(data_dir: Path | str = DATA_DIR) -> list[Document]:
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    return docs


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def retrieve(query: str, docs: list[Document] | None = None, top_k: int = 2) -> list[tuple[Document, str, int]]:
    """Return the top-k (document, chunk, overlap-score) pairs for *query*."""
    docs = docs if docs is not None else load_documents()
    query_tokens = _tokens(query)
    scored: list[tuple[int, Document, str]] = []
    for doc in docs:
        for chunk in doc.chunks:
            score = len(query_tokens & _tokens(chunk))
            if score:
                scored.append((score, doc, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].doc_id))
    return [(doc, chunk, score) for score, doc, chunk in scored[:top_k]]
