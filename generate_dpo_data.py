#!/usr/bin/env python3
"""
SupportOps v2 — DPO Preference Data Generator
=============================================
Generates a Direct Preference Optimization (DPO) dataset mapping 
conversational states in the SupportOps environment to aligned (Chosen) 
and misaligned/reward-hacked (Rejected) action responses.

Usage:
  python3 generate_dpo_data.py
  -> Output: dpo_preference_dataset.json
"""

import json
from typing import Dict, List
from env.data import TICKETS, calculate_complexity

def generate_dpo_pairs() -> List[Dict]:
    dataset = []
    
    for ticket in TICKETS:
        ticket_id = ticket["ticket_id"]
        subject = ticket["subject"]
        body = ticket["body"]
        gt = ticket["ground_truth"]
        
        # Calculate continuous difficulty
        complexity = calculate_complexity(ticket)
        
        # 1. Routing Preference Pair (Easy Task)
        correct_dept = gt["correct_department"]
        # Find a wrong department
        wrong_dept = "billing" if correct_dept != "billing" else "technical_support"
        
        prompt_route = (
            f"TICKET_ID: {ticket_id}\n"
            f"SUBJECT: {subject}\n"
            f"BODY: {body}\n"
            f"GOAL: Route the ticket to the correct department.\n"
            f"COMPLEXITY: {complexity:.3f}\n"
        )
        
        dataset.append({
            "ticket_id": ticket_id,
            "task": "route",
            "prompt": prompt_route,
            "chosen": f'{{"action_type": "route", "department": "{correct_dept}"}}',
            "rejected": f'{{"action_type": "route", "department": "{wrong_dept}"}}',
            "rationale": f"Correctly identified routing target based on key department classification rules."
        })
        
        # 2. Response Preference Pair: Aligned vs. Reward Hacking (Medium/Hard Tasks)
        key_topics = list(gt.get("key_response_topics", ["support"]))
        good_kws = list(gt.get("good_resolution_keywords", ["resolved"]))
        
        # Prompt for response
        prompt_response = (
            f"TICKET_ID: {ticket_id}\n"
            f"SUBJECT: {subject}\n"
            f"BODY: {body}\n"
            f"METADATA: Department={correct_dept}, Urgency={gt.get('correct_urgency')}\n"
            f"GOAL: Send an aligned response resolving the customer query.\n"
        )
        
        # Chosen response: Helpful, polite, structured, referencing topics
        kws_str = ", ".join(key_topics[:3])
        chosen_response = (
            f"Hello {ticket.get('sender_name', 'Customer')}, thank you for reaching out. "
            f"I have reviewed your query regarding the {kws_str} issue. "
            f"Our team is actively investigating this, and we will update you as soon as the problem is resolved. "
            f"Please let us know if you have any additional information. Best regards, Support Team."
        )
        chosen_action = {
            "action_type": "respond",
            "response_text": chosen_response
        }
        
        # Rejected response 1: Classic Reward Hacked (keyword stuffed, low judge score)
        hacked_response = " ".join(key_topics + good_kws) + " resolved solved done refund ticket support"
        rejected_action_hacking = {
            "action_type": "respond",
            "response_text": hacked_response
        }
        
        # Rejected response 2: Unhelpful / Robotic (robotic tone, no action steps)
        unhelpful_response = "Your ticket has been received. We will look at it later."
        rejected_action_unhelpful = {
            "action_type": "respond",
            "response_text": unhelpful_response
        }
        
        # Append Aligned vs Reward Hacking pair
        dataset.append({
            "ticket_id": ticket_id,
            "task": "response_alignment",
            "prompt": prompt_response,
            "chosen": json.dumps(chosen_action),
            "rejected": json.dumps(rejected_action_hacking),
            "rationale": "Mitigates reward hacking by favoring structured, polite paragraphs over raw keyword-stuffed tokens."
        })
        
        # Append Aligned vs Unhelpful pair
        dataset.append({
            "ticket_id": ticket_id,
            "task": "response_utility",
            "prompt": prompt_response,
            "chosen": json.dumps(chosen_action),
            "rejected": json.dumps(rejected_action_unhelpful),
            "rationale": "Favors helpful, actionable support responses over short, vague boilerplate messages."
        })

    return dataset

def main():
    print("=" * 60)
    print("  SupportOps DPO Preference Dataset Generator")
    print("=" * 60)
    
    pairs = generate_dpo_pairs()
    
    # Save JSON file
    output_path = "dpo_preference_dataset.json"
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
        
    print(f"\n✓ Generated {len(pairs)} preference alignment pairs.")
    print(f"✓ Saved dataset to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
