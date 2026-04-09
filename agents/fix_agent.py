import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior SRE fix recommendation agent.
Given a root cause analysis, generate actionable fix recommendations.

Rules:
- Be specific to the actual error type
- Do not give generic advice unrelated to the error
- For timeout errors: focus on latency, load, async patterns
- For memory errors: focus on heap tuning, cache limits
- For connection errors: focus on pool sizing, circuit breakers
- For disk errors: focus on log rotation, volume sizing

Return ONLY valid JSON with these exact keys:
- immediate_fix: string (action within minutes — stop the bleeding)
- short_term_fix: string (action within hours or days — stabilize)
- long_term_fix: string (architectural improvement — prevent recurrence)
- fix_summary: string (one-line summary of the fix)
Do not include markdown or explanation. Return raw JSON only."""

USER_PROMPT = """
Root Cause: {root_cause}
Service: {service_name}
Severity: {severity}
Error Type: {error_type}
Confidence: {confidence}

Generate specific fix recommendations for this exact error."""


def build_fix_agent(groq_api_key: str) -> callable:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def generate_fix(root_cause_result: dict, parsed: dict) -> dict:
        result = chain.invoke({
            "root_cause": root_cause_result.get("root_cause", ""),
            "service_name": parsed.get("service_name", "unknown"),
            "severity": parsed.get("severity", "Medium"),
            "error_type": parsed.get("error_type", ""),
            "confidence": root_cause_result.get("confidence", 0.5),
        })
        logger.info("FixAgent: fix generated for service=%s", parsed.get("service_name"))
        return result

    return generate_fix