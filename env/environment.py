"""
Support Ticket Triage — OpenEnv Environment
============================================

Implements the standard OpenEnv interface:
  reset() -> TicketObservation
  step(action: TicketAction) -> (TicketObservation, TicketReward, bool, dict)
  state() -> EnvironmentState
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from env.data import TICKET_LOOKUP, TICKETS
from env.graders import GRADERS
from env.models import (
    ActionType,
    Department,
    EnvironmentState,
    TicketAction,
    TicketMessage,
    TicketObservation,
    TicketReward,
    UrgencyLevel,
)
from env.tasks import ALL_TASKS, TASK_LOOKUP, TaskSpec

AVAILABLE_ACTIONS = [at.value for at in ActionType]
AVAILABLE_DEPARTMENTS = [d.value for d in Department]
AVAILABLE_URGENCIES = [u.value for u in UrgencyLevel]


class TicketTriageEnv:
    """
    OpenEnv-compliant environment for Support Ticket Triage.

    Args:
        task_name: One of "route", "triage", "resolve". Default: "route".
        ticket_id: Pin to a specific ticket (for reproducibility). 
                   If None, picks randomly from the task's pool.
        seed: Optional RNG seed.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        task_name: str = "route",
        ticket_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if task_name not in TASK_LOOKUP:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Choose from: {list(TASK_LOOKUP.keys())}"
            )
        self._task_spec: TaskSpec = TASK_LOOKUP[task_name]
        self._pinned_ticket_id: Optional[str] = ticket_id
        self._rng = random.Random(seed)

        # Episode state (initialised on reset)
        self._ticket_data: Dict[str, Any] = {}
        self._observation: Optional[TicketObservation] = None
        self._actions_taken: List[Dict[str, Any]] = []
        self._step_number: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._follow_up_injected: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> TicketObservation:
        """Reset the environment and return the initial observation."""
        # Pick ticket
        if self._pinned_ticket_id:
            ticket_id = self._pinned_ticket_id
        else:
            ticket_id = self._rng.choice(self._task_spec.ticket_ids)

        self._ticket_data = copy.deepcopy(TICKET_LOOKUP[ticket_id])
        self._actions_taken = []
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._follow_up_injected = False

        self._observation = TicketObservation(
            ticket_id=ticket_id,
            subject=self._ticket_data["subject"],
            body=self._ticket_data["body"],
            sender_email=self._ticket_data["sender_email"],
            sender_name=self._ticket_data["sender_name"],
            conversation_history=[
                TicketMessage(
                    sender=self._ticket_data["sender_name"],
                    content=self._ticket_data["body"],
                    timestamp=self._now(),
                )
            ],
            current_department=None,
            current_urgency=None,
            tags=[],
            is_escalated=False,
            is_closed=False,
            step_number=0,
            task_name=self._task_spec.name,
            task_description=self._task_spec.description,
            available_actions=AVAILABLE_ACTIONS,
        )
        return copy.deepcopy(self._observation)

    def step(
        self, action: TicketAction
    ) -> Tuple[TicketObservation, TicketReward, bool, Dict[str, Any]]:
        """
        Apply an action and return (observation, reward, done, info).

        Reward is shaped per-step; the terminal reward summarises the episode.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")
        if self._observation is None:
            raise RuntimeError("Call reset() before step().")

        self._step_number += 1
        self._observation.step_number = self._step_number

        step_reward, info = self._apply_action(action)
        self._actions_taken.append(action.model_dump())

        # Check terminal conditions
        done = self._check_done(action)
        self._done = done

        # On terminal step: compute final episode reward from grader
        if done:
            final_reward = self._run_grader()
            self._cumulative_reward += final_reward.value
            info["final_grader_reward"] = final_reward.model_dump()
            terminal_reward = final_reward
        else:
            # Mid-episode shaped reward
            self._cumulative_reward += step_reward.value
            terminal_reward = step_reward

        # Possibly inject a follow-up message from the customer (hard task)
        self._maybe_inject_follow_up(action)

        return copy.deepcopy(self._observation), terminal_reward, done, info

    def state(self) -> EnvironmentState:
        """Return full internal state (includes hidden ground truth for graders)."""
        if self._observation is None:
            raise RuntimeError("Call reset() before state().")
        return EnvironmentState(
            observation=copy.deepcopy(self._observation),
            ground_truth=self._ticket_data.get("ground_truth", {}),
            cumulative_reward=self._cumulative_reward,
            step_number=self._step_number,
            done=self._done,
            task_name=self._task_spec.name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(
        self, action: TicketAction
    ) -> Tuple[TicketReward, Dict[str, Any]]:
        """Mutate observation state and return a shaped step reward."""
        info: Dict[str, Any] = {}

        if action.action_type == ActionType.ROUTE:
            if action.department is None:
                return TicketReward(
                    value=0.0,
                    reason="ROUTE requires 'department' field.",
                    partial_scores={},
                ), {"error": "missing department"}
            self._observation.current_department = action.department
            gt_dept = self._ticket_data["ground_truth"]["correct_department"]
            score = 0.3 if action.department == gt_dept else 0.0
            return TicketReward(
                value=score,
                reason=f"Routed to {action.department.value}.",
                partial_scores={"routing": score},
            ), info

        elif action.action_type == ActionType.SET_URGENCY:
            if action.urgency is None:
                return TicketReward(
                    value=0.0,
                    reason="SET_URGENCY requires 'urgency' field.",
                    partial_scores={},
                ), {"error": "missing urgency"}
            self._observation.current_urgency = action.urgency
            gt_urgency = self._ticket_data["ground_truth"]["correct_urgency"]
            score = 0.2 if action.urgency == gt_urgency else 0.05
            return TicketReward(
                value=score,
                reason=f"Set urgency to {action.urgency.value}.",
                partial_scores={"urgency": score},
            ), info

        elif action.action_type == ActionType.TAG:
            if not action.tags:
                return TicketReward(
                    value=0.0,
                    reason="TAG requires non-empty 'tags' list.",
                    partial_scores={},
                ), {"error": "missing tags"}
            for tag in action.tags:
                if tag not in self._observation.tags:
                    self._observation.tags.append(tag)
            required = self._ticket_data["ground_truth"].get("required_tags", set())
            overlap = len(set(self._observation.tags) & required) / max(len(required), 1)
            return TicketReward(
                value=round(0.1 * overlap, 4),
                reason=f"Added tags: {action.tags}. Overlap with required: {overlap:.0%}",
                partial_scores={"tagging": overlap},
            ), info

        elif action.action_type == ActionType.RESPOND:
            if not action.response_text:
                return TicketReward(
                    value=0.0,
                    reason="RESPOND requires 'response_text' field.",
                    partial_scores={},
                ), {"error": "missing response_text"}
            self._observation.conversation_history.append(
                TicketMessage(
                    sender="Support Agent",
                    content=action.response_text,
                    timestamp=self._now(),
                )
            )
            key_topics = self._ticket_data["ground_truth"].get("key_response_topics", set())
            text_lower = action.response_text.lower()
            found = sum(1 for kw in key_topics if kw.lower() in text_lower)
            quality = found / max(len(key_topics), 1)
            return TicketReward(
                value=round(0.15 * quality, 4),
                reason=f"Response addresses {found}/{len(key_topics)} key topics.",
                partial_scores={"response_quality": quality},
            ), info

        elif action.action_type == ActionType.ESCALATE:
            if not action.escalation_reason:
                return TicketReward(
                    value=0.0,
                    reason="ESCALATE requires 'escalation_reason' field.",
                    partial_scores={},
                ), {"error": "missing escalation_reason"}
            self._observation.is_escalated = True
            needs = self._ticket_data["ground_truth"].get("needs_escalation", False)
            score = 0.2 if needs else -0.1  # penalise unnecessary escalation
            return TicketReward(
                value=max(0.0, score),
                reason=(
                    "Escalated correctly." if needs
                    else "Unnecessary escalation (penalised)."
                ),
                partial_scores={"escalation": score},
            ), info

        elif action.action_type == ActionType.CLOSE:
            self._observation.is_closed = True
            note = action.resolution_note or ""
            score = 0.1 if len(note.split()) >= 5 else 0.0
            return TicketReward(
                value=score,
                reason=f"Ticket closed. Resolution note length: {len(note.split())} words.",
                partial_scores={"closure": score},
            ), info

        else:  # NOOP
            return TicketReward(
                value=0.0,
                reason="NOOP action — no state change.",
                partial_scores={},
            ), info

    def _check_done(self, action: TicketAction) -> bool:
        """Episode ends on CLOSE, task-specific completion, or max_steps reached."""
        if action.action_type == ActionType.CLOSE:
            return True
        # Easy task: routing is the only required action — auto-complete after ROUTE
        if self._task_spec.name == "route" and action.action_type == ActionType.ROUTE:
            return True
        if self._step_number >= self._task_spec.max_steps:
            return True
        return False

    def _maybe_inject_follow_up(self, action: TicketAction) -> None:
        """
        For the hard task, inject a customer follow-up after the first RESPOND.
        Simulates a realistic multi-turn support conversation.
        """
        if self._task_spec.name != "resolve":
            return
        if self._follow_up_injected:
            return
        gt = self._ticket_data["ground_truth"]
        follow_up = gt.get("follow_up_message")
        if not follow_up:
            return
        respond_count = sum(
            1 for a in self._actions_taken if a.get("action_type") == ActionType.RESPOND
        )
        if respond_count >= 1:
            self._observation.conversation_history.append(
                TicketMessage(
                    sender=self._observation.sender_name,
                    content=follow_up,
                    timestamp=self._now(),
                )
            )
            self._follow_up_injected = True

    def _run_grader(self) -> TicketReward:
        """Run the task's grader on the completed episode."""
        grader_fn = GRADERS[self._task_spec.grader_name]
        episode = {
            "observation": self._observation,
            "ground_truth": self._ticket_data.get("ground_truth", {}),
            "actions_taken": self._actions_taken,
            "step_number": self._step_number,
        }
        return grader_fn(episode)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Class-level metadata
    # ------------------------------------------------------------------

    @classmethod
    def list_tasks(cls) -> List[Dict[str, str]]:
        return [
            {
                "name": t.name,
                "display_name": t.display_name,
                "difficulty": t.difficulty,
                "max_steps": str(t.max_steps),
            }
            for t in ALL_TASKS
        ]
