"""
AutoGen multi-agent setup for collaborative incident resolution.
Optional extension — not in the main LangGraph pipeline.
"""

import os
import logging
from typing import Dict

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MIN_INPUT_LENGTH = 10


def run_autogen_analysis(log_input: str) -> Dict:
    # ─────────────────────────────
    # 1. Validation
    # ─────────────────────────────
    if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
        raise ValueError("Invalid log input")

    try:
        import autogen
    except ImportError:
        raise RuntimeError("pyautogen is not installed. Run: pip install pyautogen")

    # ─────────────────────────────
    # 2. LLM Config (Groq)
    # ─────────────────────────────
    config = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
    }]

    # ─────────────────────────────
    # 3. Agents
    # ─────────────────────────────
    analyst = autogen.AssistantAgent(
        name="SRE_Analyst",
        llm_config={"config_list": config},
        system_message="""
You are a senior SRE.

Analyze the incident log and identify root cause.

Return STRICT JSON:
{
  "root_cause": "...",
  "confidence": 0.0,
  "reasoning": "..."
}

Rules:
- No hallucination
- Confidence between 0.2 and 0.95
""",
    )

    fixer = autogen.AssistantAgent(
        name="Fix_Engineer",
        llm_config={"config_list": config},
        system_message="""
You are a reliability engineer.

Based on root cause, generate fixes.

Return STRICT JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}

Rules:
- No generic advice
- Be actionable
""",
    )

    orchestrator = autogen.UserProxyAgent(
        name="Orchestrator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    # ─────────────────────────────
    # 4. Step 1 → Root Cause
    # ─────────────────────────────
    try:
        orchestrator.initiate_chat(
            analyst,
            message=f"Analyze this log:\n{log_input}"
        )
        root_output = analyst.last_message()["content"]
    except Exception as e:
        logger.error("AutoGen root cause failed: %s", e)
        return {
            "root_cause": "Failed to analyze",
            "confidence": 0.3,
            "fix_summary": "Fallback required"
        }

    # ─────────────────────────────
    # 5. Step 2 → Fix Generation
    # ─────────────────────────────
    try:
        orchestrator.initiate_chat(
            fixer,
            message=f"Root cause:\n{root_output}"
        )
        fix_output = fixer.last_message()["content"]
    except Exception as e:
        logger.error("AutoGen fix generation failed: %s", e)
        fix_output = "Fix generation failed"

    # ─────────────────────────────
    # 6. Final Output
    # ─────────────────────────────
    logger.info("AutoGen root cause: %s", root_output[:200])
    logger.info("AutoGen fix: %s", fix_output[:200])

    return {
        "root_cause_raw": root_output,
        "fix_raw": fix_output,
    }
