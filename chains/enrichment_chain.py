import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


def build_enrichment_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You enrich incident data with additional context. Given an error summary, "
            "return JSON with keys: affected_components (list), business_impact (string), "
            "urgency_score (int 1-10).",
        ),
        ("human", "Error summary: {summary}\nService: {service_name}\nSeverity: {severity}"),
    ])
    return prompt | llm | JsonOutputParser()
