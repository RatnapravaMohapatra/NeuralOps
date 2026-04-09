"""
CrewAI-based agent crew for incident analysis.
Optional extension — not in the main LangGraph pipeline.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def run_crew_analysis(log_input: str) -> str:
    try:
        from crewai import Agent, Task, Crew
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError("crewai or langchain-groq not installed.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )

    analyst = Agent(
        role="SRE Analyst",
        goal="Analyze the incident log and identify the root cause.",
        backstory="Senior SRE with 10 years of incident response experience.",
        llm=llm,
        verbose=False,
    )
    fixer = Agent(
        role="Fix Engineer",
        goal="Generate immediate and long-term fixes for the identified root cause.",
        backstory="Platform engineer specializing in resilience and reliability.",
        llm=llm,
        verbose=False,
    )

    task1 = Task(
        description=f"Analyze this log:\n{log_input}",
        agent=analyst,
        expected_output="Root cause analysis.",
    )
    task2 = Task(
        description="Generate fix recommendations based on the root cause.",
        agent=fixer,
        expected_output="Structured fix plan.",
    )

    crew = Crew(agents=[analyst, fixer], tasks=[task1, task2], verbose=False)
    result = crew.kickoff()
    return str(result)
