"""
AutoGen multi-agent setup for collaborative incident resolution.
Used as an optional extension — not in the main LangGraph pipeline.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def run_autogen_analysis(log_input: str) -> str:
    try:
        import autogen
    except ImportError:
        raise RuntimeError("pyautogen is not installed. Run: pip install pyautogen")

    config = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": "https://api.groq.com/openai/v1",
    }]

    analyst = autogen.AssistantAgent(
        name="SRE_Analyst",
        llm_config={"config_list": config},
        system_message="You are a senior SRE. Analyze the incident log and identify root cause.",
    )
    fixer = autogen.AssistantAgent(
        name="Fix_Engineer",
        llm_config={"config_list": config},
        system_message="You are a fix engineer. Given the root cause, propose a concrete remediation plan.",
    )
    user = autogen.UserProxyAgent(
        name="Orchestrator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config=False,
    )

    user.initiate_chat(analyst, message=f"Incident log:\n{log_input}")
    return analyst.last_message()["content"]
