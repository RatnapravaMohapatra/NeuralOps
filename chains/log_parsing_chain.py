import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


def build_log_parsing_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract structured data from logs.\n"
            "STRICT RULES:\n"
            "- Use only given log\n"
            "- Do not guess\n"
            "- Return valid JSON only\n\n"
            "Keys:\n"
            "error_type, service_name, severity (Critical/High/Medium/Low), summary"
        ),
        ("human", "{log_input}"),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def run(log_input: str) -> dict:
        try:
            result = chain.invoke({"log_input": log_input})
        except Exception as e:
            logger.error("Parsing failed: %s", e)
            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": "Medium",
                "summary": log_input[:200],
            }

        result = result or {}

        return {
            "error_type": result.get("error_type", "UnknownError"),
            "service_name": result.get("service_name", "unknown"),
            "severity": result.get("severity", "Medium"),
            "summary": result.get("summary", log_input[:200]),
        }

    return run
