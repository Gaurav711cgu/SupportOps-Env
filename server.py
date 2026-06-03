"""
FastAPI server — Support Ticket Triage OpenEnv
================================================

Exposes the standard OpenEnv HTTP API:
  GET  /           → healthcheck + metadata
  POST /reset      → start episode, returns observation
  POST /step       → apply action, returns (obs, reward, done, info)
  GET  /state      → full internal state (with ground truth)
  GET  /tasks      → list all tasks
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from env.environment import TicketTriageEnv
from env.models import ActionType, Department, TicketAction, UrgencyLevel

app = FastAPI(
    title="Support Ticket Triage — OpenEnv",
    description=(
        "An OpenEnv environment where AI agents must triage, route, and resolve "
        "customer support tickets. Real-world task with 3 difficulty levels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (keyed by session_id)
# ---------------------------------------------------------------------------

_sessions: Dict[str, TicketTriageEnv] = {}


def _get_env(session_id: str) -> TicketTriageEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "route"
    ticket_id: Optional[str] = None
    seed: Optional[int] = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    department: Optional[str] = None
    response_text: Optional[str] = None
    urgency: Optional[str] = None
    tags: Optional[list[str]] = None
    escalation_reason: Optional[str] = None
    resolution_note: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def healthcheck():
    return {
        "status": "ok",
        "environment": "Support Ticket Triage",
        "version": "1.0.0",
        "tasks": TicketTriageEnv.list_tasks(),
    }


@app.get("/tasks")
def list_tasks():
    return {"tasks": TicketTriageEnv.list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = req.session_id or str(uuid.uuid4())
    env = TicketTriageEnv(
        task_name=req.task_name,
        ticket_id=req.ticket_id,
        seed=req.seed,
    )
    _sessions[session_id] = env
    obs = env.reset()
    return {"observation": obs.model_dump(), "session_id": session_id}


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.session_id)

    # Parse action
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{req.action_type}'. "
                   f"Valid: {[a.value for a in ActionType]}",
        )

    dept = None
    if req.department:
        try:
            dept = Department(req.department)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid department '{req.department}'.",
            )

    urg = None
    if req.urgency:
        try:
            urg = UrgencyLevel(req.urgency)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid urgency '{req.urgency}'.",
            )

    action = TicketAction(
        action_type=action_type,
        department=dept,
        response_text=req.response_text,
        urgency=urg,
        tags=req.tags,
        escalation_reason=req.escalation_reason,
        resolution_note=req.resolution_note,
    )

    obs, reward, done, info = env.step(action)

    if done:
        # Clean up session after episode ends
        _sessions.pop(req.session_id, None)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
        session_id=req.session_id,
    )


@app.get("/state")
def state(session_id: str):
    env = _get_env(session_id)
    s = env.state()
    return s.model_dump()


@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    html_path = os.path.join(os.path.dirname(__file__), "env", "ui.html")
    try:
        with open(html_path, "r") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UI template load error: {e}")


# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
