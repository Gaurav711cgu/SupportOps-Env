"""
Graders for the Support Ticket Triage OpenEnv environment.

All graders are deterministic and produce scores in [0.0, 1.0].
Each grader receives the full episode state and returns a TicketReward.
"""

from __future__ import annotations

from typing import Any, Dict, List

from env.models import ActionType, Department, TicketReward, UrgencyLevel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _keyword_overlap(text: str, keywords: set, threshold: int = 1) -> float:
    """
    Returns fraction of keywords found in text (case-insensitive).
    If text contains >= threshold keywords → full credit per keyword found.
    """
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def _tag_overlap(actual_tags: List[str], required_tags: set) -> float:
    """Fraction of required tags that appear in actual_tags."""
    if not required_tags:
        return 1.0
    actual_set = {t.lower() for t in actual_tags}
    required_lower = {t.lower() for t in required_tags}
    found = actual_set & required_lower
    return len(found) / len(required_lower)


_JUDGE_API_DISABLED = False

def llm_judge_score(response: str, ticket: dict) -> float:
    """
    Score response quality 0.0-1.0 using Gemini as judge.
    Falls back to a robust heuristic-based grader if API call fails or key is missing.
    """
    global _JUDGE_API_DISABLED
    import os
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None

    # 1. Attempt real API call if keys are present and not disabled
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key and OpenAI and not _JUDGE_API_DISABLED:
        try:
            # Let's configure base url for Gemini or OpenAI
            if api_key.startswith("AIzaSy"):
                base_url = os.getenv("ANTHROPIC_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
                if "openai" not in base_url and "generativelanguage.googleapis.com" in base_url:
                    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                model = "gemini-2.0-flash"
            else:
                base_url = "https://api.openai.com/v1"
                model = "gpt-4o-mini"

            client = OpenAI(base_url=base_url, api_key=api_key)
            rubric = f"""
            Ticket Subject: {ticket.get('subject', '')}
            Ticket Body: {ticket.get('body', '')[:200]}
            Agent Response: {response}

            Score 0.0–1.0 on:
            - Addresses the customer's actual problem (0.4 weight)
            - Professional tone without being robotic (0.3 weight)
            - Provides actionable next steps (0.3 weight)

            Return ONLY a single float number between 0.0 and 1.0 (e.g. 0.85).
            """

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a customer service grader. Output only a float between 0.0 and 1.0."},
                    {"role": "user", "content": rubric}
                ],
                temperature=0.0,
                max_tokens=10,
                timeout=5.0  # Limit timeout for the judge call
            )
            text_result = completion.choices[0].message.content.strip()
            return float(text_result)
        except Exception:
            _JUDGE_API_DISABLED = True

    # 2. Heuristic fallback grader
    # Evaluate length, customer service tone, and reward hacking (keyword stuffing)
    if not response or len(response.strip()) == 0:
        return 0.0

    words = response.lower().split()
    total_words = len(words)

    gt = ticket.get("ground_truth", {})
    key_topics = gt.get("key_response_topics", set())
    follow_up_topics = gt.get("follow_up_response_topics", set())
    all_kws = set(list(key_topics) + list(follow_up_topics))

    kw_hits = [w for w in words if any(kw.lower() in w for kw in all_kws)]

    # Keyword stuffing detection
    if total_words < 10 and len(kw_hits) > 0 and len(kw_hits) / total_words > 0.5:
        return 0.15

    if total_words > 0:
        rep_ratio = len(kw_hits) / total_words
        if rep_ratio > 0.6:  # Over 60% of response is keywords
            return 0.20

    score = 0.0

    # Word count length checks
    if 20 <= total_words <= 120:
        score += 0.3
    elif 10 <= total_words < 20:
        score += 0.15
    elif total_words > 120:
        score += 0.2

    # Tone/greetings checks
    has_greeting = any(g in response.lower() for g in ["hi", "hello", "dear", "thank you", "thanks"])
    has_closing = any(c in response.lower() for c in ["best", "regards", "sincerely", "support team", "help"])
    if has_greeting:
        score += 0.15
    if has_closing:
        score += 0.15

    # Politeness and actionability checks
    polite = any(p in response.lower() for p in ["sorry", "apologize", "please", "glad to", "happy to", "assist"])
    action = any(a in response.lower() for a in ["will", "should", "steps", "link", "click", "resolve", "update", "fixed", "check", "refund"])
    if polite:
        score += 0.2
    if action:
        score += 0.2

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Route Grader (Easy task)
# ---------------------------------------------------------------------------

def route_grader(episode: Dict[str, Any]) -> TicketReward:
    """
    Score: 1.0 if routed to the correct department, else 0.0.
    Small partial credit (0.1) for attempting a route at all vs. noop-ing.
    """
    ground_truth: Dict = episode["ground_truth"]
    actions_taken: List[Dict] = episode.get("actions_taken", [])
    observation = episode["observation"]

    correct_dept: Department = ground_truth["correct_department"]
    current_dept = observation.current_department

    # Did the agent ever perform a ROUTE action?
    route_actions = [a for a in actions_taken if a.get("action_type") == ActionType.ROUTE]

    if not route_actions:
        return TicketReward(
            value=0.0,
            reason="No ROUTE action taken.",
            partial_scores={"routing": 0.0},
        )

    if current_dept == correct_dept:
        return TicketReward(
            value=1.0,
            reason=f"Correctly routed to {correct_dept.value}.",
            partial_scores={"routing": 1.0},
        )

    # Routed but to the wrong department
    # Give 0.1 partial credit for at least routing (vs. doing nothing)
    return TicketReward(
        value=0.1,
        reason=(
            f"Routed to {current_dept.value if current_dept else 'unknown'} "
            f"but correct answer is {correct_dept.value}."
        ),
        partial_scores={"routing": 0.1},
    )


# ---------------------------------------------------------------------------
# Triage Grader (Medium task)
# ---------------------------------------------------------------------------

def triage_grader(episode: Dict[str, Any]) -> TicketReward:
    """
    Weighted score across four sub-tasks:
      - Routing:   0.30
      - Urgency:   0.25
      - Tagging:   0.20
      - Response:  0.25
    """
    ground_truth: Dict = episode["ground_truth"]
    observation = episode["observation"]
    actions_taken: List[Dict] = episode.get("actions_taken", [])

    # --- 1. Routing (0.30) ---
    correct_dept: Department = ground_truth["correct_department"]
    routing_score = 1.0 if observation.current_department == correct_dept else 0.0

    # --- 2. Urgency (0.25) ---
    correct_urgency: UrgencyLevel = ground_truth["correct_urgency"]
    urgency_score = 0.0
    if observation.current_urgency == correct_urgency:
        urgency_score = 1.0
    elif observation.current_urgency is not None:
        # Partial credit for adjacent urgency levels
        levels = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
        correct_idx = levels.index(correct_urgency)
        actual_idx = levels.index(observation.current_urgency)
        diff = abs(correct_idx - actual_idx)
        urgency_score = max(0.0, 1.0 - diff * 0.4)

    # --- 3. Tagging (0.20) ---
    required_tags: set = ground_truth.get("required_tags", set())
    tagging_score = _tag_overlap(observation.tags, required_tags)

    # --- 4. Response quality (0.25) ---
    key_topics: set = ground_truth.get("key_response_topics", set())
    response_texts = [
        a.get("response_text", "") or ""
        for a in actions_taken
        if a.get("action_type") == ActionType.RESPOND
    ]
    combined_response = " ".join(response_texts)
    
    # Dual-signal grader: 50% keyword overlap, 50% LLM judge score
    keyword_score = min(1.0, _keyword_overlap(combined_response, key_topics) * 1.5)
    ticket_context = {
        "subject": observation.subject,
        "body": observation.body,
        "ground_truth": ground_truth,
    }
    judge_score = llm_judge_score(combined_response, ticket_context)
    response_score = 0.5 * keyword_score + 0.5 * judge_score

    # Aggregate
    weights = {"routing": 0.30, "urgency": 0.25, "tagging": 0.20, "response": 0.25}
    partial = {
        "routing": routing_score,
        "urgency": urgency_score,
        "tagging": tagging_score,
        "response": response_score,
    }
    total = sum(weights[k] * v for k, v in partial.items())
    total = round(min(1.0, max(0.0, total)), 4)

    return TicketReward(
        value=total,
        reason=(
            f"Routing={'✓' if routing_score == 1 else '✗'} "
            f"Urgency={'✓' if urgency_score == 1 else f'{urgency_score:.2f}'} "
            f"Tags={tagging_score:.2f} Response={response_score:.2f}"
        ),
        partial_scores=partial,
    )


# ---------------------------------------------------------------------------
# Resolve Grader (Hard task)
# ---------------------------------------------------------------------------

def resolve_grader(episode: Dict[str, Any]) -> TicketReward:
    """
    Weighted score across six sub-tasks:
      - Routing:      0.15
      - Urgency:      0.10
      - Initial resp: 0.20
      - Escalation:   0.20
      - Follow-up:    0.20
      - Closure:      0.15
    """
    ground_truth: Dict = episode["ground_truth"]
    observation = episode["observation"]
    actions_taken: List[Dict] = episode.get("actions_taken", [])

    # --- 1. Routing (0.15) ---
    correct_dept: Department = ground_truth["correct_department"]
    routing_score = 1.0 if observation.current_department == correct_dept else 0.0

    # --- 2. Urgency (0.10) ---
    correct_urgency: UrgencyLevel = ground_truth["correct_urgency"]
    levels = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
    if observation.current_urgency == correct_urgency:
        urgency_score = 1.0
    elif observation.current_urgency is not None:
        diff = abs(levels.index(correct_urgency) - levels.index(observation.current_urgency))
        urgency_score = max(0.0, 1.0 - diff * 0.4)
    else:
        urgency_score = 0.0

    # --- 3. Initial response quality (0.20) ---
    key_topics: set = ground_truth.get("key_response_topics", set())
    respond_actions = [a for a in actions_taken if a.get("action_type") == ActionType.RESPOND]
    initial_response = respond_actions[0].get("response_text", "") if respond_actions else ""
    
    # Dual-signal grader: 50% keyword, 50% LLM judge
    keyword_score = min(1.0, _keyword_overlap(initial_response, key_topics) * 1.5)
    ticket_context = {
        "subject": observation.subject,
        "body": observation.body,
        "ground_truth": ground_truth,
    }
    judge_score = llm_judge_score(initial_response, ticket_context)
    initial_resp_score = 0.5 * keyword_score + 0.5 * judge_score

    # --- 4. Escalation (0.20) ---
    needs_escalation: bool = ground_truth.get("needs_escalation", False)
    escalate_actions = [a for a in actions_taken if a.get("action_type") == ActionType.ESCALATE]
    if needs_escalation:
        if escalate_actions:
            reason = escalate_actions[0].get("escalation_reason", "") or ""
            escalation_score = 0.5 + 0.5 * min(1.0, len(reason.split()) / 10.0)
        else:
            escalation_score = 0.0
    else:
        # Should NOT have escalated
        escalation_score = 0.0 if escalate_actions else 1.0

    # --- 5. Follow-up response (0.20) ---
    follow_up_topics: set = ground_truth.get("follow_up_response_topics", set())
    follow_up_response = respond_actions[1].get("response_text", "") if len(respond_actions) > 1 else ""
    if follow_up_topics:
        # Dual-signal grader: 50% keyword, 50% LLM judge
        keyword_score = min(1.0, _keyword_overlap(follow_up_response, follow_up_topics) * 1.5)
        ticket_context = {
            "subject": observation.subject,
            "body": observation.body,
            "ground_truth": ground_truth,
        }
        judge_score = llm_judge_score(follow_up_response, ticket_context)
        follow_up_score = 0.5 * keyword_score + 0.5 * judge_score
    else:
        follow_up_score = 1.0  # No follow-up expected → full credit

    # --- 6. Closure quality (0.15) ---
    close_actions = [a for a in actions_taken if a.get("action_type") == ActionType.CLOSE]
    if not close_actions:
        closure_score = 0.0
    else:
        resolution_note = close_actions[0].get("resolution_note", "") or ""
        good_keywords: set = ground_truth.get("good_resolution_keywords", set())
        closure_score = (
            min(1.0, _keyword_overlap(resolution_note, good_keywords) * 1.5)
            if good_keywords else (1.0 if resolution_note else 0.3)
        )

    weights = {
        "routing": 0.15,
        "urgency": 0.10,
        "initial_response": 0.20,
        "escalation": 0.20,
        "follow_up": 0.20,
        "closure": 0.15,
    }
    partial = {
        "routing": routing_score,
        "urgency": urgency_score,
        "initial_response": initial_resp_score,
        "escalation": escalation_score,
        "follow_up": follow_up_score,
        "closure": closure_score,
    }
    total = sum(weights[k] * v for k, v in partial.items())
    total = round(min(1.0, max(0.0, total)), 4)

    return TicketReward(
        value=total,
        reason=" | ".join(f"{k}={v:.2f}" for k, v in partial.items()),
        partial_scores=partial,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS = {
    "route_grader": route_grader,
    "triage_grader": triage_grader,
    "resolve_grader": resolve_grader,
}
