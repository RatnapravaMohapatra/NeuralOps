import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

def build_enrichment_chain():
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Return JSON with keys: affected_components, business_impact, urgency_score (1-10)"),
        ("human", "Error: {summary}\nService: {service_name}\nSeverity: {severity}")
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def run(summary, service_name, severity):
        try:
            result = chain.invoke({
                "summary": summary,
                "service_name": service_name,
                "severity": severity
            })
        except Exception:
            return {
                "affected_components": [],
                "business_impact": "Unknown",
                "urgency_score": 5
            }

        return {
            "affected_components": result.get("affected_components", []),
            "business_impact": result.get("business_impact", "Unknown"),
            "urgency_score": int(result.get("urgency_score", 5))
        }

    return run
