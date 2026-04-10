import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior SRE root cause analysis agent.

STRICT RULES:
1. Use ONLY explicit evidence from the log.
2. Follow the evidence table strictly.
3. DO NOT guess beyond given signals.

EVIDENCE TABLE:

"timed out" + URL → Service slow/overloaded → 0.75
"timed out" no URL → Internal processing slow → 0.70
"ConnectionRefused" → Service down → 0.85
"500 Internal" → Application bug → 0.80
"OOM" → Memory issue → 0.90
"pool exhausted" → DB pool full → 0.90
"certificate" → SSL issue → 0.92
"no space left" → Disk full → 0.95
"Insufficient cpu" → K8s resource issue → 0.88
"429" → Rate limit → 0.90
"NullPointerException" → Code bug → 0.80

Return ONLY JSON:
- root_cause
- confidence
- reasoning
"""

USER_PROMPT = """
Log:
{summary}

Error Type: {error_type}
Service: {service_name}
Severity: {severity}

Context:
{rag_context}
"""


def build_root_cause_agent(groq_api_key: str) -> callable:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def analyze(parsed: dict, rag_results: list[dict]) -> dict:

        rag_context = "\n".join(
            f"- {r.get('root_cause','')}"
            for r in rag_results
        ) or "No context"

        try:
            result = chain.invoke({
                "summary": parsed.get("summary", ""),
                "error_type": parsed.get("error_type", ""),
                "service_name": parsed.get("service_name", ""),
                "severity": parsed.get("severity", ""),
                "rag_context": rag_context,
            })

        except Exception as e:
            logger.error("RootCause failed: %s", str(e))

            # ✅ SAFE FALLBACK
            return {
                "root_cause": "Unable to determine root cause due to processing failure.",
                "confidence": 0.3,
                "reasoning": "LLM pipeline failed"
            }

        # ✅ SAFETY FIXES
        result = result or {}

        result.setdefault("root_cause", "Insufficient evidence.")
        result.setdefault("confidence", 0.5)
        result.setdefault("reasoning", "No clear signal found.")

        # clamp confidence
        try:
            conf = float(result["confidence"])
        except:
            conf = 0.5

        result["confidence"] = round(max(0.0, min(1.0, conf)), 3)

        logger.info("RootCauseAgent: confidence=%.3f", result["confidence"])

        return result

    return analyze
