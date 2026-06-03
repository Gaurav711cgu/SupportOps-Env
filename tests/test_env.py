import pytest
from env.data import TICKETS, calculate_complexity
from env.environment import TicketTriageEnv
from env.models import ActionType, Department, TicketAction, UrgencyLevel, TicketObservation
from env.graders import route_grader, triage_grader, resolve_grader, llm_judge_score

def test_ticket_complexity():
    """Verify that continuous complexity scores are bounded in [0, 1] and match expectations."""
    for ticket in TICKETS:
        c = calculate_complexity(ticket)
        assert 0.0 <= c <= 1.0, f"Complexity {c} out of bounds for ticket {ticket['ticket_id']}"
    
    # Assert specific complexity ordering
    # TKT-001 (Easy, 1 turn) should have lower complexity than TKT-008 (Hard data loss, multi-turn)
    t1 = next(t for t in TICKETS if t["ticket_id"] == "TKT-001")
    t8 = next(t for t in TICKETS if t["ticket_id"] == "TKT-008")
    assert calculate_complexity(t1) < calculate_complexity(t8)

def test_environment_reset_and_step():
    """Test standard environment MDP state transitions (reset, step, constraints)."""
    env = TicketTriageEnv(task_name="route", ticket_id="TKT-001", seed=42)
    obs = env.reset()
    
    assert obs.ticket_id == "TKT-001"
    assert obs.current_department is None
    assert obs.step_number == 0
    assert not obs.is_closed
    
    # Take an action
    action = TicketAction(action_type=ActionType.ROUTE, department=Department.BILLING)
    obs, reward, done, info = env.step(action)
    
    assert obs.current_department == Department.BILLING
    assert obs.step_number == 1
    # ROUTE task should end immediately after a ROUTE action or max steps
    assert done
    assert env._cumulative_reward > 0.0

def test_reward_hacking_detection():
    """Verify that the grader correctly identifies and penalizes keyword stuffing (reward hacking)."""
    ticket = next(t for t in TICKETS if t["ticket_id"] == "TKT-001")
    gt = ticket["ground_truth"]
    key_topics = gt.get("key_response_topics", set())
    
    # 1. Aligned, polite paragraph response
    polite_text = (
        "Hi Jane, thank you for reaching out. We apologize for the double billing. "
        "I have processed a refund of the extra amount to your card. Please let us "
        "know if you need further help."
    )
    score_polite = llm_judge_score(polite_text, {"ground_truth": gt})
    
    # 2. Reward hacked: bare list of keywords repeated to exceed 60% density
    hacked_text = " ".join(list(key_topics)) + " "
    hacked_text = hacked_text * 10  # e.g., "refund charge apologize refund charge..."
    score_hacked = llm_judge_score(hacked_text, {"ground_truth": gt})
    
    # The polite, coherent text should score much higher than the hacked text
    assert score_polite > 0.7
    assert score_hacked <= 0.20

def test_graders_score_boundaries():
    """Ensure all core graders map final episode scores cleanly to the [0.0, 1.0] range."""
    # Test route grader
    obs = TicketObservation(
        ticket_id="TKT-001",
        subject="Test",
        body="Test",
        sender_email="test@test.com",
        sender_name="Test",
        current_department=Department.BILLING
    )
    episode_route = {
        "ground_truth": {"correct_department": Department.BILLING},
        "actions_taken": [{"action_type": ActionType.ROUTE}],
        "observation": obs
    }
    r = route_grader(episode_route)
    assert 0.0 <= r.value <= 1.0
