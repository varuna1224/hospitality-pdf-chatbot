"""
llm.py
------
STEP 7 of the RAG pipeline: LLM MODELS (Generation)

Purpose : Generates a context-grounded, hospitality-toned answer using
          the retrieved chunks + the user's question.

Supports either OpenAI or Anthropic as the generation backend -
set LLM_PROVIDER in your environment (.env file) to "openai" or
"anthropic". Only ONE key is required, whichever provider you pick.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()   # "openai" | "anthropic"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a courteous hospitality assistant for a hotel/resort.
Answer ONLY using the CONTEXT provided below, which comes from the
property's own PDF documents (tariff sheets, SOPs, guest policies, menus,
banquet packages, amenities lists, etc.).

Rules:
1. If the answer is not in the context, say clearly:
   "I don't have that information in the uploaded documents. Please check with the front desk."
2. Keep the tone warm, professional, and concise - like a helpful hotel concierge.
3. Never invent prices, dates, room numbers, or policies that aren't in the context.
4. When useful, mention which document the info came from.
"""


def _call_openai(question: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nGUEST QUESTION:\n{question}"},
        ],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_anthropic(question: str, context: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nGUEST QUESTION:\n{question}"}
        ],
    )
    return response.content[0].text


def generate_answer(question: str, context: str) -> str:
    """
    Routes the generation call to whichever provider is configured.
    """
    try:
        if LLM_PROVIDER == "anthropic":
            return _call_anthropic(question, context)
        return _call_openai(question, context)
    except Exception as e:
        return (
            "⚠️ I couldn't reach the language model right now "
            f"({e}). Please check your API key / LLM_PROVIDER setting in .env."
        )
