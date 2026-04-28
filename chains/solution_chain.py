import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


def build_solution_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
    )

    SYSTEM_PROMPT = """
You are a senior SRE.

Generate actionable fixes.

RULES:
- NO generic advice
- Provide real steps

Return JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
Root Cause: {root_cause}
Service: {service_name}
Severity: {severity}
""")
    ])

    chain = prompt | llm | JsonOutputParser()

    def run(data: Dict[str, Any]) -> Dict:
        try:
            result = chain.invoke(data)

            return {
                "immediate_fix": result.get("immediate_fix", "Restart service"),
                "short_term_fix": result.get("short_term_fix", "Check logs"),
                "long_term_fix": result.get("long_term_fix", "Improve monitoring"),
                "fix_summary": result.get("fix_summary", "General fix applied"),
            }

        except Exception as e:
            logger.error("Solution failed: %s", e)
            return {
                "immediate_fix": "Restart service",
                "short_term_fix": "Check logs",
                "long_term_fix": "Improve monitoring",
                "fix_summary": "Fallback solution",
            }

    return run
