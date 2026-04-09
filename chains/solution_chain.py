import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

def build_solution_chain():
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate structured SRE fixes.\n"
            "Return JSON:\n"
            "immediate_fix, short_term_fix, long_term_fix, fix_summary"
        ),
        ("human", "Root cause: {root_cause}\nService: {service_name}\nSeverity: {severity}"),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def run(root_cause, service_name, severity):
        try:
            result = chain.invoke({
                "root_cause": root_cause,
                "service_name": service_name,
                "severity": severity
            })
        except Exception:
            return {
                "immediate_fix": "Restart service",
                "short_term_fix": "Investigate logs",
                "long_term_fix": "Improve monitoring",
                "fix_summary": "Fallback response"
            }

        return {
            "immediate_fix": result.get("immediate_fix", ""),
            "short_term_fix": result.get("short_term_fix", ""),
            "long_term_fix": result.get("long_term_fix", ""),
            "fix_summary": result.get("fix_summary", "")
        }

    return run
