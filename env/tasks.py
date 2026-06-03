"""
Task definitions for the Support Ticket Triage OpenEnv environment.

Each task specifies:
  - name & description shown to the agent
  - which tickets belong to this task
  - max_steps allowed
  - grader function reference
  - expected difficulty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskSpec:
    name: str
    display_name: str
    description: str
    ticket_ids: List[str]
    max_steps: int
    difficulty: str          # "easy" | "medium" | "hard"
    grader_name: str         # matches key in graders.py GRADERS dict
    success_threshold: float  # minimum score considered "solved"


# ---------------------------------------------------------------------------
# Task 1 — EASY: Route to the correct department
# ---------------------------------------------------------------------------
TASK_ROUTE = TaskSpec(
    name="route",
    display_name="Ticket Routing (Easy)",
    description=(
        "Read the support ticket and route it to the correct department by "
        "using the ROUTE action. You have one chance to route correctly. "
        "Available departments: billing, technical_support, sales, "
        "customer_success, legal. "
        "Score 1.0 for the correct department, 0.0 for wrong."
    ),
    ticket_ids=["TKT-001", "TKT-002", "TKT-003", "TKT-004", "TKT-005"],
    max_steps=3,
    difficulty="easy",
    grader_name="route_grader",
    success_threshold=1.0,
)


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: Route + Set Urgency + Tag + Respond
# ---------------------------------------------------------------------------
TASK_TRIAGE = TaskSpec(
    name="triage",
    display_name="Full Triage (Medium)",
    description=(
        "Fully triage the support ticket by completing ALL of the following: "
        "(1) ROUTE to the correct department, "
        "(2) SET_URGENCY to the appropriate level (low/medium/high/critical), "
        "(3) TAG with relevant classification labels, "
        "(4) RESPOND with a helpful initial reply to the customer. "
        "You must close the ticket with CLOSE after completing all steps. "
        "Partial credit is awarded for each sub-task completed correctly."
    ),
    ticket_ids=["TKT-006", "TKT-007", "TKT-001", "TKT-003"],
    max_steps=8,
    difficulty="medium",
    grader_name="triage_grader",
    success_threshold=0.7,
)


# ---------------------------------------------------------------------------
# Task 3 — HARD: Full resolution including follow-up and escalation
# ---------------------------------------------------------------------------
TASK_RESOLVE = TaskSpec(
    name="resolve",
    display_name="Full Resolution (Hard)",
    description=(
        "Fully resolve a complex support ticket over multiple steps. "
        "You must: (1) ROUTE correctly, (2) SET_URGENCY appropriately, "
        "(3) RESPOND with an empathetic and actionable initial reply, "
        "(4) ESCALATE with a clear reason if the ticket warrants it, "
        "(5) Handle a follow-up message from the customer with another RESPOND, "
        "(6) CLOSE the ticket with a resolution summary. "
        "All steps are required for full score. Graders check response quality, "
        "escalation decisions, and resolution completeness."
    ),
    ticket_ids=["TKT-008", "TKT-009"],
    max_steps=12,
    difficulty="hard",
    grader_name="resolve_grader",
    success_threshold=0.6,
)


ALL_TASKS: List[TaskSpec] = [TASK_ROUTE, TASK_TRIAGE, TASK_RESOLVE]
TASK_LOOKUP = {t.name: t for t in ALL_TASKS}
