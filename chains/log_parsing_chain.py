import os
import logging
import time
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 2000


def _sanitize(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:MAX_INPUT_LENGTH]


def _safe_output(result: Any, fallback: str) -> Dict:
    if not isinstance(result, dict):
        result = {}

    return {
        "error_type": result.get("error_type") or "UnknownError",
        "service_name": result.get("service_name") or "unknown",
        "severity": result.get("severity") or "Medium",
        "summary": result.get("summary") or fallback[:200],
    }


def build_log_parsing_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )

    SYSTEM_PROMPT = """
You are an expert SRE log parser.

Extract structured data from logs.

RULES:
- NEVER leave fields empty
- If missing → infer conservatively
- DO NOT hallucinate technologies

FIELDS:
- error_type: main error keyword
- service_name: infer if possible, else "unknown"
- severity: Critical / High / Medium / Low
- summary: ALWAYS provide meaningful summary

Return ONLY JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "...",
  "summary": "..."
}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{log_input}")
    ])

    chain = prompt | llm | JsonOutputParser()

    def parse(data: Dict[str, Any]) -> Dict:
        log_input = _sanitize(data.get("log_input", ""))

        if len(log_input) < MIN_INPUT_LENGTH:
            raise ValueError("Invalid log input")

        try:
            result = chain.invoke({"log_input": log_input})
            return _safe_output(result, log_input)

        except Exception as e:
            logger.error("Parsing failed: %s", e)
            return _safe_output({}, log_input)

    return parse
