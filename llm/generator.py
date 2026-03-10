"""
RAG SYSTEM — STEP 5: LLM Answer Generation
============================================
Takes retrieved chunks + user question → generates a grounded answer.

The key RAG principle: the LLM is NOT generating from memory.
It is reading the retrieved documents and summarizing/extracting
the answer from them. This prevents hallucination.

Chain of Thought used here:
    1. Retrieve relevant chunks  (done by retriever.py)
    2. Format them as context
    3. Send context + question to LLM
    4. LLM extracts the answer from the context
    5. If answer not in context, say so — don't hallucinate

Language handling:
    - Detects user language (EN/DE) from the question
    - Responds in the SAME language as the question
    - Context may be in a different language — that's OK,
      the LLM can read German context and answer in English
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_groq import ChatGroq

load_dotenv()

# ── Config ────────────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.2"))  # Low = more factual

# How many context characters to send to LLM
# ~8000 chars ≈ ~2000 tokens of context
MAX_CONTEXT_CHARS = 8000

# ── LLM client (singleton) ───────────────────────────────────

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY not set in environment.\n"
                "Get a free key at https://console.groq.com"
            )
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            groq_api_key=GROQ_API_KEY,
        )
    return _llm


# ── Language Detection ────────────────────────────────────────

GERMAN_WORDS = {
    "ich", "bitte", "danke", "möchte", "termin", "wie", "kann",
    "was", "wann", "wo", "haben", "ist", "sind", "kommen", "buchen",
    "fragen", "hilfe", "bitte", "nein", "ja", "und", "oder", "nicht",
    "ein", "eine", "der", "die", "das", "mit", "für", "bei",
    "öffnungszeiten", "arzt", "ärztin", "behandlung",
}


def detect_language(text: str) -> str:
    """
    Detect if text is German or English.
    Returns 'de' or 'en'.
    Simple heuristic — good enough for clinic use case.
    """
    words = set(re.findall(r"\b\w+\b", text.lower()))
    german_hits = words & GERMAN_WORDS
    # Also check for German-specific characters
    has_umlauts = bool(re.search(r"[äöüßÄÖÜ]", text))

    if len(german_hits) >= 2 or has_umlauts:
        return "de"
    return "en"


# ── Context Builder ───────────────────────────────────────────

def build_context(chunks: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Each chunk is prefixed with its source page name so the LLM
    can attribute information correctly.
    """
    context_parts = []
    total_chars   = 0

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("page_name") or chunk.metadata.get("title", f"Source {i+1}")
        text   = chunk.page_content.strip()

        part = f"[{source}]\n{text}"

        if total_chars + len(part) > max_chars:
            # Truncate this chunk to fit
            remaining = max_chars - total_chars - len(f"[{source}]\n")
            if remaining > 100:
                part = f"[{source}]\n{text[:remaining]}..."
                context_parts.append(part)
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n---\n\n".join(context_parts)


# ── System Prompts ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional AI assistant for Functiomed, a medical clinic in Zurich, Switzerland.

CRITICAL RULES:
1. Answer ONLY from the provided document context — never from general knowledge about medicine or other clinics.
2. Detect the language of the user's question. If German → respond in German. If English → respond in English.
3. If the answer is clearly present in the context (even in a different language), extract and present it.
4. If the answer is NOT in the context, say exactly: 
   - English: "I don't have that information available. Please contact Functiomed directly."
   - German: "Diese Information liegt mir nicht vor. Bitte kontaktieren Sie Functiomed direkt."
5. Never invent names, phone numbers, prices, or medical advice.
6. Keep answers concise and conversational — you may be speaking to someone via voice.
7. If the context contains contact info (phone/email), include it in the answer when relevant.

You help with:
- Clinic services and treatments
- Appointment booking information  
- Opening hours and location
- Staff and practitioners
- Pricing and insurance questions"""


def generate_answer(
    question: str,
    chunks: list[Document],
    language: Optional[str] = None,
    voice_mode: bool = False,
) -> str:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        question   — user's question
        chunks     — retrieved document chunks from retriever.py
        language   — 'en' or 'de' (auto-detected if None)
        voice_mode — if True, strip markdown for TTS output

    Returns:
        Answer string in the user's language.
    """
    if not chunks:
        lang = language or detect_language(question)
        if lang == "de":
            return "Es tut mir leid, ich konnte keine relevanten Informationen finden. Bitte kontaktieren Sie Functiomed direkt."
        return "I'm sorry, I couldn't find relevant information. Please contact Functiomed directly."

    # Build context from chunks
    context = build_context(chunks)
    lang    = language or detect_language(question)

    # Build the prompt
    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION ({lang.upper()}):
{question}

Answer in {'German' if lang == 'de' else 'English'} based strictly on the context above:"""

    llm = get_llm()

    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ])
        answer = response.content.strip()

        # Clean up for voice if needed
        if voice_mode:
            answer = _clean_for_voice(answer)

        return answer

    except Exception as e:
        print(f"❌ LLM error: {e}")
        if lang == "de":
            return "Entschuldigung, ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut."
        return "Sorry, an error occurred. Please try again."


def _clean_for_voice(text: str) -> str:
    """
    Remove markdown formatting for TTS.
    TTS reads '**bold**' as 'asterisk asterisk bold asterisk asterisk' — bad.
    """
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bullet points — replace with natural pause
    text = re.sub(r"^[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Full RAG Pipeline (convenience function) ─────────────────

def ask(
    question: str,
    retriever=None,
    top_n: int = 8,
    voice_mode: bool = False,
) -> dict:
    """
    One-shot RAG: retrieve + generate answer.

    Args:
        question  — user's question
        retriever — HybridRetriever instance (from retrieval/retriever.py)
        top_n     — how many chunks to retrieve
        voice_mode — clean output for TTS

    Returns:
        {
            "question": str,
            "answer": str,
            "language": "en" | "de",
            "sources": [{"title": str, "url": str}],
            "chunks_used": int,
        }
    """
    from retrieval.retriever import get_retriever

    r = retriever or get_retriever()

    # Retrieve relevant chunks
    chunks = r.retrieve(question, top_n=top_n)

    # Detect language
    lang = detect_language(question)

    # Generate answer
    answer = generate_answer(question, chunks, language=lang, voice_mode=voice_mode)

    # Collect unique sources
    seen_sources = set()
    sources = []
    for chunk in chunks:
        url   = chunk.metadata.get("source", "")
        title = chunk.metadata.get("page_name") or chunk.metadata.get("title", "")
        key   = url or title
        if key and key not in seen_sources:
            seen_sources.add(key)
            sources.append({"title": title, "url": url})

    return {
        "question":    question,
        "answer":      answer,
        "language":    lang,
        "sources":     sources[:5],   # max 5 sources in response
        "chunks_used": len(chunks),
    }


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ingest.chunker import load_documents_from_disk, chunk_documents
    from retrieval.retriever import HybridRetriever

    print("Loading RAG pipeline...\n")
    docs      = load_documents_from_disk()
    chunks    = chunk_documents(docs)
    retriever = HybridRetriever(chunks)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ").strip()

    result = ask(question, retriever=retriever)

    print(f"\n{'='*60}")
    print(f"Q: {result['question']}")
    print(f"Language: {result['language']}")
    print(f"{'='*60}")
    print(f"\n{result['answer']}\n")
    print(f"Sources ({result['chunks_used']} chunks):")
    for s in result["sources"]:
        print(f"  • {s['title']} — {s['url']}")
    print()