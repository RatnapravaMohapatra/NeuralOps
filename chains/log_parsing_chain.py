import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


def build_log_parsing_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract structured data from logs. Return JSON only with keys: "
            "error_type, service_name, severity (Critical/High/Medium/Low), summary.",
        ),
        ("human", "{log_input}"),
    ])
    return prompt | llm | JsonOutputParser()
