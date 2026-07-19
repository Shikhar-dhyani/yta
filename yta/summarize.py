"""Summary add-on: answer the question from retrieved segments via Gemini."""

from . import config

_PROMPT = """\
You are answering a question using excerpts from YouTube video transcripts.

Question: {question}

Transcript excerpts (each with video title and timestamp):
{context}

Write a concise answer (3-6 sentences) to the question based ONLY on these
excerpts. Cite the video title and timestamp (e.g. "Video Title @ 12:34") for
each claim. If the excerpts don't actually answer the question, say so plainly.
"""


def summarize(question: str, results: list[dict]) -> str:
    """Generate a Gemini answer grounded in the search results.

    Raises RuntimeError when no API key is configured.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env to enable summaries."
        )
    if not results:
        return "No matching transcript segments found in the database."

    from google import genai

    context = "\n\n".join(
        f"[{r['video_title']} @ {r['timestamp']}]\n{r['text']}" for r in results
    )
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=_PROMPT.format(question=question, context=context),
    )
    return (response.text or "").strip()
