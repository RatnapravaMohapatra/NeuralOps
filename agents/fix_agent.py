import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior SRE fix recommendation agent with 10+ years of production experience.

GOLDEN RULE: Every fix must reference the SPECIFIC error in this incident.
Never produce generic advice that could apply to any system.

════════════════════════════════════════
BANNED PHRASES — never use these
════════════════════════════════════════
"monitor your system"
"check the logs"
"optimize your application"
"improve performance"
"ensure proper configuration"
"review your setup"

Instead → name the exact tool, config parameter, or pattern.

════════════════════════════════════════
ERROR → SPECIFIC FIX MAPPING
════════════════════════════════════════

pool exhausted:
  immediate  → Increase HikariCP maximumPoolSize (e.g. from 10 → 20) and restart service
  short_term → Run EXPLAIN ANALYZE on slowest queries, set connectionTimeout=30000ms,
               ensure connections are released in finally{} blocks
  long_term  → Add Prometheus metrics for pool utilization, implement read replicas,
               add Redis caching layer to reduce direct DB load

timed out + URL (service slow/overloaded):
  immediate  → Check CPU/memory of target service (kubectl top pods), restart if unresponsive,
               temporarily increase client timeout for validation
  short_term → Profile the slow endpoint with async tracing (Jaeger/Zipkin),
               identify blocking calls, implement async processing or request queue
  long_term  → Add HPA autoscaling for target service, implement circuit breaker (Resilience4j),
               optimize inference pipeline (batching, model quantization)

timed out no URL (internal blocking):
  immediate  → Take thread dump (jstack PID), identify stuck threads, kill/restart blocked process
  short_term → Add timeout guards around all external calls, increase thread pool size,
               use CompletableFuture for async patterns
  long_term  → Refactor blocking I/O to async/reactive patterns, add request timeout middleware,
               set per-operation timeouts (DB query timeout, HTTP client timeout)

OOM / heap exhausted:
  immediate  → Restart JVM service immediately, temporarily increase -Xmx (e.g. -Xmx6g)
  short_term → Run heap dump analysis: jmap -dump:format=b,file=heap.hprof PID,
               identify leak with Eclipse MAT or VisualVM, cap cache with Caffeine eviction policy
  long_term  → Implement bounded cache (Caffeine maximumSize=10000, expireAfterWrite=1h),
               add JVM heap alerting at 80%, tune GC with -XX:+UseG1GC

Connection refused (service down):
  immediate  → Check service status: kubectl get pods -n production | grep [service],
               restart pod: kubectl rollout restart deployment/[service]
  short_term → Add readiness/liveness probes in K8s deployment spec,
               check upstream dependency health, verify environment config
  long_term  → Implement service mesh (Istio) with automatic failover,
               add circuit breaker pattern, improve zero-downtime deployment strategy

disk full:
  immediate  → Free space now: find /var/log -name "*.log" -mtime +7 -delete,
               check largest files: du -sh /var/log/* | sort -rh | head -10
  short_term → Fix logrotate: /etc/logrotate.d/app with daily, compress, rotate 7, maxsize 100M,
               add disk usage alert at 80% (Prometheus node_filesystem_avail_bytes)
  long_term  → Mount dedicated /var/log volume with autoscaling,
               implement centralized logging (ELK/Loki), enforce log retention policy

TLS / certificate expired:
  immediate  → Renew manually: certbot renew --cert-name [domain], reload nginx/ingress
  short_term → Install cert-manager: helm install cert-manager jetstack/cert-manager,
               create ClusterIssuer with Let's Encrypt for automated renewal
  long_term  → Add certificate expiry monitoring: alert 30 days before expiry,
               document renewal runbook, rotate secrets via Vault

rate limit / 429:
  immediate  → Back off immediately: implement exponential backoff with jitter,
               reduce request rate by 50% temporarily
  short_term → Add token bucket rate limiter (Guava RateLimiter or bucket4j),
               cache upstream API responses (TTL=60s), distribute requests over time
  long_term  → Request quota increase from provider, implement request queuing with priority,
               add multi-tenant rate limiting at API gateway level

K8s resource exhaustion / FailedScheduling:
  immediate  → kubectl describe node [node] to check pressure,
               cordon overloaded node: kubectl cordon [node], drain and reschedule
  short_term → Set resource requests and limits on all pods (CPU/memory),
               enable cluster autoscaler: --enable-cluster-autoscaler
  long_term  → Right-size pods using VPA (Vertical Pod Autoscaler),
               implement resource quotas per namespace, add node autoscaling policy

NullPointerException:
  immediate  → Identify null reference from full stack trace line number,
               add null check as emergency hotfix and redeploy
  short_term → Add input validation at service boundary, write unit tests for null edge cases,
               use Optional<T> pattern for nullable values
  long_term  → Enforce null safety via static analysis (SpotBugs, SonarQube in CI pipeline),
               adopt defensive programming standards across codebase

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
Return ONLY valid JSON — no markdown, no explanation:
{
  "immediate_fix": "specific action within minutes — name exact tool or command",
  "short_term_fix": "specific action within hours or days — name exact config or pattern",
  "long_term_fix": "architectural change to prevent recurrence — name exact technology",
  "fix_summary": "one sentence: what is being fixed and how"
}"""

USER_PROMPT = """
Root Cause: {root_cause}
Evidence from log: {evidence}
Service: {service_name}
Severity: {severity}
Error Type: {error_type}
Confidence: {confidence}

Generate fixes specific to this exact error and evidence.
Reference the evidence in your fix recommendations where relevant.
No generic advice."""


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

    def generate_fix(root_cause_result: dict, parsed: dict) -> dict:
        try:
            result = chain.invoke({
                "root_cause": root_cause_result.get("root_cause", ""),
                "evidence": root_cause_result.get("evidence", ""),
                "service_name": parsed.get("service_name", "unknown"),
                "severity": parsed.get("severity", "Medium"),
                "error_type": parsed.get("error_type", ""),
                "confidence": root_cause_result.get("confidence", 0.5),
            })
        except Exception as e:
            logger.error("FixAgent failed: %s", e)
            return {
                "immediate_fix": "Check service logs and restart if unresponsive.",
                "short_term_fix": "Investigate root cause using profiler or distributed tracing.",
                "long_term_fix": "Add monitoring and alerting for this failure pattern.",
                "fix_summary": "Manual investigation required — automated fix unavailable.",
            }

        result = result or {}
        result.setdefault("immediate_fix", "Check service health immediately.")
        result.setdefault("short_term_fix", "Investigate root cause and apply targeted patch.")
        result.setdefault("long_term_fix", "Add monitoring and prevention strategy.")
        result.setdefault("fix_summary", "See fix details above.")

        logger.info("FixAgent: done for service=%s", parsed.get("service_name"))
        return result

    return generate_fix
