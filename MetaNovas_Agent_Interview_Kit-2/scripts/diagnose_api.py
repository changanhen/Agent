"""Local-only API/first-step diagnostics. Never prints keys or raw responses.

python -m scripts.diagnose_api          # one SDK smoke request
python -m scripts.diagnose_api --agent  # first agent decision, no business tools executed
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import ssl
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import interview_sdk as sdk


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=True), flush=True)


def error_info(exc):
    """Return only fixed categories; never expose provider error body text."""
    text = str(exc)
    status = re.search(r"^Provider returned HTTP (\d{3})", text)
    code = int(status.group(1)) if status else None
    categories = {400: "request_rejected", 401: "authentication_failed",
                  402: "balance_or_payment_required", 403: "access_denied",
                  404: "model_or_endpoint_not_found", 429: "rate_limit_or_quota"}
    category = categories.get(code, "provider_error" if code else "unknown_error")
    if text == "Set INTERVIEW_API_KEY for local practice":
        category = "missing_api_key"
    elif text == "Local practice supports openai or deepseek":
        category = "unsupported_provider"
    elif isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError)):
        reason = getattr(exc, "reason", exc)
        category = "tls_error" if isinstance(reason, ssl.SSLError) else "connection_or_timeout"
    elif isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        category = "response_or_local_format_error"
    elif code and code >= 500:
        category = "provider_server_error"
    # Report only known parameter names, never any original error text.
    param = None
    if code == 400:
        for name in ("max_completion_tokens", "max_tokens", "temperature", "reasoning_effort", "thinking"):
            if name in text:
                param = name
                break
    return {"category": category, "http_status": code, "parameter_hint": param}


def describe_response(raw):
    """Metadata only: do not print business records or private reasoning."""
    try:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]
        reason = choice.get("finish_reason")
        allowed = {"stop", "length", "tool_calls", "content_filter", "insufficient_system_resource"}
        emit("http_response", finish_reason=reason if isinstance(reason, str) and reason in allowed else "other",
             content_present=bool(message.get("content")),
             reasoning_present=bool(message.get("reasoning_content")),
             tool_calls_present=bool(message.get("tool_calls")))
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        emit("http_response", category="unexpected_response_structure")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="store_true", help="Trace the first agent decision only")
    args = parser.parse_args(argv)
    provider = os.getenv("INTERVIEW_PROVIDER", "openai").strip().lower()
    emit("configuration", provider=provider if provider in sdk.PROVIDERS else "unsupported",
         key_present=bool(os.getenv("INTERVIEW_API_KEY")),
         model_override_present=bool(os.getenv("INTERVIEW_MODEL")),
         model_name_has_spaces=" " in os.getenv("INTERVIEW_MODEL", ""),
         proxy_present=bool(os.getenv("MODEL_PROXY_SOCKET")))
    if not os.getenv("INTERVIEW_API_KEY") and not os.getenv("MODEL_PROXY_SOCKET"):
        emit("error", category="missing_api_key")
        return 2

    real_open = sdk.urllib.request.urlopen

    @contextmanager
    def observed_open(*positional, **keywords):
        with real_open(*positional, **keywords) as response:
            raw = response.read()
        describe_response(raw)
        with io.BytesIO(raw) as buffered:
            yield buffered

    def traced_chat(*positional, **keywords):
        emit("model_request", stage="candidate" if keywords.get("tools") else "review")
        try:
            return sdk.chat(*positional, **keywords)
        except Exception as exc:
            emit("error", **error_info(exc))
            raise

    try:
        with patch.object(sdk.urllib.request, "urlopen", observed_open):
            if not args.agent:
                emit("model_request", stage="api_smoke_test")
                answer = sdk.chat([{"role": "user", "content": "Reply with OK only."}], max_tokens=4096)
                ok = isinstance(answer, dict) and bool(answer.get("content") or answer.get("tool_call"))
                emit("api_result", usable_output=ok)
                return 0 if ok else 1

            import agent
            from scripts.practice_runner import POLICY, TASK, TOOLS

            info = TASK["user_scenario"]["instructions"]
            messages = [{"role": "user", "content": info["known_info"] + "\n\n" + info["reason_for_call"]}]
            real_review = agent.review_candidate

            def traced_review(*positional, **keywords):
                report = real_review(*positional, **keywords)
                checks = report.get("checks", {})
                emit("audit", approved=report.get("approved") is True,
                     issue_count=len(report.get("issues", [])),
                     checks_present=isinstance(checks, dict) and bool(checks),
                     failed_checks=[key for key in ("preconditions", "latest_intent", "evidence",
                                    "confirmation", "batch_progress", "completion_claims")
                                    if not isinstance(checks, dict) or checks.get(key) is not True])
                return report

            with patch.object(agent, "chat", traced_chat), patch.object(agent, "review_candidate", traced_review):
                answer = agent.respond(messages, TOOLS, POLICY)
            kind = "tool_call" if "tool_call" in answer else "content"
            fallback = answer.get("content", "").startswith("I couldn't verify the information needed")
            emit("agent_result", response_kind=kind, fallback=fallback, business_tools_executed=0)
            return 1 if fallback else 0
    except Exception as exc:
        emit("error", **error_info(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
