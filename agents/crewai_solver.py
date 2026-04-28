"""
CrewAI-based agent crew for incident analysis.
Optional extension — not in the main LangGraph pipeline.
"""

import os
import logging
from typing import Dict

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MIN_INPUT_LENGTH = 10


def run_crew_analysis(log_input: str) -> Dict:
    # ─────────────────────────────
    # 1. Validation
    # ─────────────────────────────
    if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
        raise ValueError("Invalid log input")

    try:
        from crewai import Agent, Task, Crew
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError("crewai or langchain-groq not installed.")

    # ─────────────────────────────
    # 2. LLM Setup
    # ─────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0,
    )

    # ─────────────────────────────
    # 3. Agents
    # ─────────────────────────────
    analyst = Agent(
        role="Senior SRE Analyst",
        goal="Identify precise root cause from logs without hallucination.",
        backstory="Expert in debugging distributed systems, logs, and production incidents.",
        llm=llm,
        verbose=False,
    )

    fixer = Agent(
        role="Fix Engineer",
        goal="Provide concrete, actionable fixes based on root cause.",
        backstory="Specialist in system recovery, DevOps, and reliability engineering.",
        llm=llm,
        verbose=False,
    )

    # ─────────────────────────────
    # 4. Tasks (Structured)
    # ─────────────────────────────
    task1 = Task(
        description=f"""
Analyze the following log carefully:

{log_input}

Return STRICT JSON:
{{
  "root_cause": "...",
  "confidence": 0.0,
  "reasoning": "..."
}}

Rules:
- No guessing
- Confidence between 0.2 and 0.95
""",
        agent=analyst,
        expected_output="Structured root cause JSON",
    )

    task2 = Task(
        description="""
Using the previous root cause analysis, generate fixes.

Return STRICT JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}

Rules:
- No generic advice
- Include commands or actions if possible
""",
        agent=fixer,
        expected_output="Structured fix JSON",
    )

    # ─────────────────────────────
    # 5. Crew Execution
    # ─────────────────────────────
    crew = Crew(
        agents=[analyst, fixer],
        tasks=[task1, task2],
        verbose=False,
    )

    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error("CrewAI execution failed: %s", e)
        return {
            "root_cause": "CrewAI failed to analyze",
            "confidence": 0.3,
            "fix_summary": "Fallback fix required",
        }

    # ─────────────────────────────
    # 6. Output Handling
    # ─────────────────────────────
    output = str(result)

    logger.info("CrewAI result: %s", output[:300])

    # ⚠️ CrewAI returns string → keep simple parse fallback
    return {
        "raw_output": output
    }
