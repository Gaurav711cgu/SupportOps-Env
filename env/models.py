"""
Typed Pydantic models for the Support Ticket Triage OpenEnv environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------

class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    SALES = "sales"
    CUSTOMER_SUCCESS = "customer_success"
    LEGAL = "legal"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    ROUTE = "route"           # Assign to a department
    RESPOND = "respond"       # Send a reply to the customer
    SET_URGENCY = "set_urgency"  # Set urgency level
    TAG = "tag"               # Add classification tags
    ESCALATE = "escalate"     # Escalate with a reason
    CLOSE = "close"           # Resolve and close the ticket
    NOOP = "noop"             # Do nothing (wastes a step)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class TicketAction(BaseModel):
    """One action the agent can take in the environment."""

    action_type: ActionType = Field(
        description="The type of action to perform."
    )
    department: Optional[Department] = Field(
        default=None,
        description="Target department (required for ROUTE action).",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Message body sent to the customer (required for RESPOND).",
    )
    urgency: Optional[UrgencyLevel] = Field(
        default=None,
        description="Urgency level (required for SET_URGENCY).",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Classification tags to apply (required for TAG).",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Plain-text reason for escalation (required for ESCALATE).",
    )
    resolution_note: Optional[str] = Field(
        default=None,
        description="Summary of how the issue was resolved (required for CLOSE).",
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class TicketMessage(BaseModel):
    """One message in the ticket conversation thread."""

    sender: str
    content: str
    timestamp: str


class TicketObservation(BaseModel):
    """Everything the agent can observe at a given step."""

    # Ticket content
    ticket_id: str
    subject: str
    body: str
    sender_email: str
    sender_name: str

    # Evolving state
    conversation_history: List[TicketMessage] = Field(default_factory=list)
    current_department: Optional[Department] = None
    current_urgency: Optional[UrgencyLevel] = None
    tags: List[str] = Field(default_factory=list)
    is_escalated: bool = False
    is_closed: bool = False

    # Episode metadata
    step_number: int = 0
    task_name: str = ""
    task_description: str = ""
    available_actions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class TicketReward(BaseModel):
    """Structured reward with partial-credit breakdown."""

    value: float = Field(ge=0.0, le=1.0, description="Aggregate reward [0, 1].")
    reason: str = Field(description="Human-readable explanation.")
    partial_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion scores contributing to value.",
    )


# ---------------------------------------------------------------------------
# State model (returned by state())
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state snapshot (superset of observation)."""

    observation: TicketObservation
    ground_truth: Dict[str, Any] = Field(
        description="Hidden ground-truth labels used by graders.",
    )
    cumulative_reward: float = 0.0
    step_number: int = 0
    done: bool = False
    task_name: str = ""
