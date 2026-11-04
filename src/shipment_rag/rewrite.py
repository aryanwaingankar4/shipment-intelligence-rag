"""Turn a question into a standalone search question, using the recent conversation.

"What about the second one?" can't be searched for. A small LLM call replaces it with what it
refers to, and everything after that works as it does for a question asked on its own. The answer
model never sees the conversation, so it can still only answer from the retrieved passages.

Never fatal: if the model fails, or its rewrite adds a shipment ID or product code nobody
mentioned, the question is used as written.
"""

import json
import logging
import re

log = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "You rewrite questions for a search over a pharmaceutical logistics company's documents. "
    "You never answer them."
)
REWRITE_PROMPT = """Conversation so far:
{conversation}

Latest question: {question}

Rewrite the latest question as one standalone question for searching the documents.
- Replace words like "it", "that one" or "the second one" with what they refer to in the
  conversation. Copy shipment IDs and product codes exactly.
- Use the formal wording a company document would use instead of casual wording.
- If it is already complete or starts a new topic, keep its meaning and ignore the conversation.
- Do not answer it. Do not add any fact, number, name or ID that is not in the question or the
  conversation."""
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

_ID = re.compile(r"\b(?:SH-\d{4}-\d{3}|TL-[A-Z]+-\d+)\b", re.IGNORECASE)
_CITATION = re.compile(r"\s*\[\d+\]")
MAX_REWRITE_CHARS = 500
MAX_MESSAGE_CHARS = 500  # a long answer is cut: the rewriter only needs what it was about


def format_conversation(history: list[dict], max_messages: int) -> str:
    lines = []
    for message in history[-max_messages:] if max_messages > 0 else []:
        speaker = "User" if message["role"] == "user" else "Assistant"
        text = _CITATION.sub("", message["content"]).strip()
        lines.append(f"{speaker}: {text[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines) or "(none)"


def _invented_id(rewrite: str, question: str, conversation: str) -> str | None:
    known = {m.upper() for m in _ID.findall(f"{question}\n{conversation}")}
    return next((m for m in _ID.findall(rewrite) if m.upper() not in known), None)


class QueryRewriter:
    def __init__(self, llm, max_messages: int = 6):
        self.llm = llm
        self.max_messages = max_messages

    def rewrite(self, question: str, history: list[dict] | None = None) -> str:
        """The standalone form of the question, or the question itself if that isn't possible."""
        conversation = format_conversation(history or [], self.max_messages)
        try:
            raw = self.llm.generate(
                REWRITE_SYSTEM,
                REWRITE_PROMPT.format(conversation=conversation, question=question.strip()),
                json_schema=REWRITE_SCHEMA,
            )
            rewrite = " ".join(json.loads(raw)["question"].split())
        except Exception as exc:  # a failed rewrite must never cost the user their answer
            log.warning("query rewrite failed, using the question as written",
                        extra={"error": type(exc).__name__})
            return question
        if not rewrite or len(rewrite) > MAX_REWRITE_CHARS:
            log.warning("query rewrite unusable, using the question as written")
            return question
        invented = _invented_id(rewrite, question, conversation)
        if invented:
            log.warning("query rewrite added an unknown ID, using the question as written",
                        extra={"id": invented})
            return question
        return rewrite
