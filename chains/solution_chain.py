import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


def build_solution_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate SRE fix recommendations. Return JSON with keys: "
            "immediate_fix, short_term_fix, long_term_fix, fix_summary.",
        ),
        ("human", "Root cause: {root_cause}\nService: {service_name}\nSeverity: {severity}"),
    ])
    return prompt | llm | JsonOutputParser()
