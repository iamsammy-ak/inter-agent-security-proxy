"""FastAPI service: an OpenAI-compatible chat endpoint, an inspection endpoint and an agent-to-agent relay.

Callers identify themselves with ``X-Agent-Id`` and, when the policy stores a key hash for that agent,
``Authorization: Bearer <key>``. ``X-Session-Id`` groups the agents of one workflow so they share PII tokens.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__
from .audit import VerifyResult
from .llm import Upstream, UpstreamError
from .models import SCOPE_AUDIT_READ, PermissionDenied, Principal
from .proxy import InjectionBlocked, InspectReport, ProxyError, RelayRequest, RelayResult, SecurityProxy


class InspectRequest(BaseModel):
    text: str
    session_id: str = "inspect"


class RevealRequest(BaseModel):
    session_id: str
    text: str


class RevealResponse(BaseModel):
    text: str


def create_app(proxy: SecurityProxy, upstream: Upstream) -> FastAPI:
    """Build the ASGI app around an existing proxy and upstream (tests pass a mock upstream)."""
    app = FastAPI(title="Inter-Agent Security Proxy", version=__version__,
                  description="PII tokenisation, injection defence, least-privilege tool policy and audit logging "
                              "for messages passed between agents.")

    def principal(
        x_agent_id: str = Header(..., description="calling agent"),
        authorization: str | None = Header(None),
    ) -> Principal:
        bearer = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
        found = proxy.engine.authenticate(x_agent_id, bearer)
        if found is None:
            proxy.audit.append("auth.failed", actor=x_agent_id)
            raise HTTPException(status_code=401, detail="unknown agent or wrong key")
        return found

    @app.exception_handler(ProxyError)
    async def _proxy_error(_: Request, exc: ProxyError) -> JSONResponse:
        content: dict[str, Any] = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        if isinstance(exc, InjectionBlocked):
            content["error"]["verdict"] = exc.verdict.model_dump()
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        classifier = proxy.detector.classifier
        return {"status": "ok", "version": __version__, "upstream": upstream.name,
                "classifier": classifier.name if classifier else None, "agents": sorted(proxy.policy.agents)}

    @app.post("/v1/inspect", response_model=InspectReport)
    def inspect(body: InspectRequest, caller: Principal = Depends(principal)) -> InspectReport:
        return proxy.inspect(body.text, session_id=body.session_id, actor=caller.agent_id)

    @app.post("/v1/chat/completions")
    def chat_completions(
        payload: dict[str, Any],
        caller: Principal = Depends(principal),
        x_session_id: str = Header("default"),
    ) -> dict[str, Any]:
        try:
            return proxy.chat(caller.agent_id, x_session_id, payload, upstream)
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/relay", response_model=RelayResult)
    def relay(body: RelayRequest, caller: Principal = Depends(principal)) -> Any:
        if body.sender != caller.agent_id:
            proxy.audit.append("relay.denied", actor=caller.agent_id, session=body.session_id,
                               reasons=[f"caller {caller.agent_id} cannot send as {body.sender}"])
            raise HTTPException(status_code=403, detail="sender must be the authenticated agent")
        result = proxy.relay(body)
        if not result.delivered:  # 422: blocked as an injection; 403: refused by policy
            status = 422 if result.verdict.is_injection else 403
            return JSONResponse(status_code=status, content=result.model_dump())
        return result

    @app.post("/v1/reveal", response_model=RevealResponse)
    def reveal(body: RevealRequest, caller: Principal = Depends(principal)) -> RevealResponse:
        try:
            return RevealResponse(text=proxy.reveal(body.session_id, body.text, caller))
        except PermissionDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/v1/audit/verify", response_model=VerifyResult)
    def verify(caller: Principal = Depends(principal)) -> VerifyResult:
        if not caller.has_scope(SCOPE_AUDIT_READ):
            raise HTTPException(status_code=403, detail=f"{SCOPE_AUDIT_READ} scope required")
        return proxy.audit.verify()

    return app
