import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# =========================================================
# SYSTEM PROMPT (YOUR FULL VERSION + FIXES)
# =========================================================
SYSTEM_PROMPT = """You are a senior SRE fix recommendation agent with 10+ years of production experience.

GOLDEN RULE:
Every fix MUST reference the SPECIFIC error evidence provided.

MANDATORY:
- You MUST reference the provided evidence explicitly in at least one fix
- If no mapping matches → use root cause text only (DO NOT GUESS)
- If evidence is weak → say "Insufficient evidence from log"

════════════════════════════════════════
BANNED PHRASES — never use these
════════════════════════════════════════
"monitor your system"
"check the logs"
"optimize your application"
"improve performance"
"ensure proper configuration"
"review your setup"

════════════════════════════════════════
ERROR → SPECIFIC FIX MAPPING
════════════════════════════════════════

pool exhausted:
  immediate  → Increase HikariCP maximumPoolSize (e.g. from 10 → 20) and restart service
  short_term → Run EXPLAIN ANALYZE on slowest queries, set connectionTimeout=30000ms,
               ensure connections are released in finally{} blocks
  long_term  → Add Prometheus metrics, Redis caching, read replicas

timed out + URL:
  immediate  → kubectl top pods, restart service, increase timeout temporarily
  short_term → Use Jaeger/Zipkin tracing, identify blocking calls
  long_term  → Add autoscaling (HPA), circuit breaker (Resilience4j)

timed out no URL:
  immediate  → Take thread dump (jstack PID), restart blocked process
  short_term → Add timeout guards, increase thread pool
  long_term  → Move to async architecture

OOM / heap exhausted:
  immediate  → Restart JVM, increase -Xmx
  short_term → Analyze heap dump (jmap + MAT)
  long_term  → Add bounded cache, tune GC

Connection refused:
  immediate  → kubectl rollout restart deployment/[service]
  short_term → Add readiness/liveness probes
  long_term  → Add service mesh (Istio), circuit breaker

disk full:
  immediate  → Delete old logs
  short_term → Configure logrotate
  long_term  → Centralized logging (ELK/Loki)

TLS / certificate:
  immediate  → certbot renew
  short_term → setup cert-manager
  long_term  → add expiry monitoring

rate limit / 429:
  immediate  → exponential backoff
  short_term → token bucket limiter
  long_term  → request quota increase

NullPointerException:
  immediate  → add null check at failing line
  short_term → input validation + tests
  long_term  → static analysis (SonarQube)

════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON)
════════════════════════════════════════
{
  "immediate_fix": "specific actionable fix",
  "short_term_fix": "targeted improvement",
  "long_term_fix": "architectural prevention",
  "fix_summary": "one-line summary"
}
"""

# =========================================================
# USER PROMPT
# =========================================================
USER_PROMPT = """
Root Cause: {root_cause}
Evidence: {evidence}
Service: {service_name}
Severity: {severity}
Error Type: {error_type}
Confidence: {confidence}

Generate fixes strictly based on evidence.
"""

# =========================================================
# BUILD AGENT
# =========================================================
def build_fix_agent(groq_api_key: str) -> callable:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # =========================================================
    # MAIN FUNCTION
    # =========================================================
    def generate_fix(root_cause_result: dict, parsed: dict) -> dict:
        try:
            # ✅ FIXED: ensure evidence always present
            evidence = (
                root_cause_result.get("evidence")
                or parsed.get("summary")
                or "Insufficient evidence from log"
            )

            result = chain.invoke({
                "root_cause": root_cause_result.get("root_cause", ""),
                "evidence": evidence,
                "service_name": parsed.get("service_name", "unknown"),
                "severity": parsed.get("severity", "Medium"),
                "error_type": parsed.get("error_type", ""),
                "confidence": root_cause_result.get("confidence", 0.5),
            })

        except Exception as e:
            logger.error("FixAgent failed: %s", e)

            # ✅ FIXED: non-generic fallback
            return {
                "immediate_fix": f"Restart {parsed.get('service_name', 'service')} and inspect error: {parsed.get('summary', '')[:100]}",
                "short_term_fix": "Reproduce issue with same input and enable debug logging for failing component.",
                "long_term_fix": "Add structured monitoring and alerting for this failure pattern.",
                "fix_summary": "Fallback fix generated due to LLM failure.",
            }

        result = result or {}

        # ✅ ensure consistent output
        return {
            "immediate_fix": result.get("immediate_fix", ""),
            "short_term_fix": result.get("short_term_fix", ""),
            "long_term_fix": result.get("long_term_fix", ""),
            "fix_summary": result.get("fix_summary", ""),
        }

    return generate_fix
