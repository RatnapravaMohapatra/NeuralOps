import os
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


def build_root_cause_agent(api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )

    SYSTEM_PROMPT = """
You are a senior Site Reliability Engineer.

Identify the MOST LIKELY root cause.

RULES:
- NEVER return "Unknown cause"
- If uncertain → give best hypothesis
- Use common failure patterns:
  timeout, memory, connection pool, config, network

Return JSON:
{
  "root_cause": "...",
  "confidence": 0.0-1.0,
  "reasoning": "..."
}

Confidence:
- 0.8+ → clear
- 0.6–0.8 → likely
- 0.3–0.6 → uncertain guess
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
Parsed Log:
{parsed}

Similar Incidents:
{rag}
""")
    ])

    chain = prompt | llm | JsonOutputParser()

    def run(parsed: Dict, rag: List, retry_count=0, prev_confidence=0.0):
        try:
            result = chain.invoke({
                "parsed": parsed,
                "rag": rag or "No similar incidents"
            })

            if not result.get("root_cause"):
                result["root_cause"] = "Likely system issue due to insufficient data"

            if result.get("confidence", 0) < 0.3:
                result["confidence"] = 0.3

            return result

        except Exception as e:
            logger.error("Root cause failed: %s", e)
            return {
                "root_cause": "Likely system issue due to insufficient data",
                "confidence": 0.3,
                "reasoning": "fallback"
            }

    return run
