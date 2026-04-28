import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

def build_enrichment_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You enrich incident data.

Return JSON:
{
  "affected_components": [],
  "business_impact": "",
  "urgency_score": 1
}

Rules:
- urgency_score must be 1–10
- No null values
"""),
        ("human", "Summary: {summary}\nService: {service_name}\nSeverity: {severity}")
    ])

    chain = prompt | llm | JsonOutputParser()

    def enrich(data: dict):
        try:
            result = chain.invoke(data)
        except Exception as e:
            logger.error("Enrichment failed: %s", e)
            return {
                "affected_components": [],
                "business_impact": "Unknown impact",
                "urgency_score": 5,
            }

        if not isinstance(result, dict):
            result = {}

        result["affected_components"] = result.get("affected_components") or []
        result["business_impact"] = result.get("business_impact") or "Unknown"
        
        try:
            score = int(result.get("urgency_score", 5))
        except:
            score = 5

        result["urgency_score"] = max(1, min(10, score))

        return result

    return enrich
