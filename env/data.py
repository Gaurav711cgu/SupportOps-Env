"""
Curated synthetic support tickets with ground-truth labels.

Each ticket is a dict with:
  - ticket fields (subject, body, sender_*)
  - ground_truth:
      correct_department   : Department enum value
      correct_urgency      : UrgencyLevel enum value
      required_tags        : set of tags graders accept
      key_response_topics  : keywords a good response must address
      needs_escalation     : bool
"""

from env.models import Department, UrgencyLevel

TICKETS = [
    # -----------------------------------------------------------------------
    # EASY routing tickets — one clear signal per email
    # -----------------------------------------------------------------------
    {
        "ticket_id": "TKT-001",
        "subject": "Double charged on my invoice #4821",
        "body": (
            "Hi, I was charged twice for my subscription this month. "
            "My invoice number is #4821. The duplicate charge appeared on "
            "March 15th. Please refund the extra amount ASAP. "
            "My account email is jane.smith@example.com."
        ),
        "sender_email": "jane.smith@example.com",
        "sender_name": "Jane Smith",
        "ground_truth": {
            "correct_department": Department.BILLING,
            "correct_urgency": UrgencyLevel.HIGH,
            "required_tags": {"billing", "duplicate-charge", "refund"},
            "key_response_topics": {"refund", "invoice", "charge", "apologize"},
            "needs_escalation": False,
            "good_resolution_keywords": {"refund processed", "resolved", "credited"},
        },
    },
    {
        "ticket_id": "TKT-002",
        "subject": "API returning 500 errors since yesterday",
        "body": (
            "Our production environment is broken. Your API has been returning "
            "HTTP 500 errors on the /v2/users endpoint since yesterday 6pm UTC. "
            "This is blocking our entire checkout flow. Error code: INTERNAL_500_USR. "
            "We need this fixed urgently."
        ),
        "sender_email": "ops@startup.io",
        "sender_name": "DevOps Team",
        "ground_truth": {
            "correct_department": Department.TECHNICAL_SUPPORT,
            "correct_urgency": UrgencyLevel.CRITICAL,
            "required_tags": {"api", "500-error", "production-outage"},
            "key_response_topics": {"error", "investigating", "update", "escalate"},
            "needs_escalation": True,
            "good_resolution_keywords": {"resolved", "fix deployed", "restored"},
        },
    },
    {
        "ticket_id": "TKT-003",
        "subject": "Interested in enterprise pricing for 500 seats",
        "body": (
            "Hello, we are evaluating your platform for our company of ~500 people. "
            "Could you share enterprise pricing, volume discounts, and whether you "
            "offer annual contracts? We'd also love a demo call with your team."
        ),
        "sender_email": "procurement@bigcorp.com",
        "sender_name": "Sarah Johnson",
        "ground_truth": {
            "correct_department": Department.SALES,
            "correct_urgency": UrgencyLevel.MEDIUM,
            "required_tags": {"enterprise", "pricing", "demo-request"},
            "key_response_topics": {"pricing", "demo", "enterprise", "contact"},
            "needs_escalation": False,
            "good_resolution_keywords": {"demo scheduled", "pricing sent", "follow-up"},
        },
    },
    {
        "ticket_id": "TKT-004",
        "subject": "Need help setting up SSO with Okta",
        "body": (
            "We are trying to configure SAML SSO with Okta but keep getting "
            "'Assertion validation failed' errors. We have followed the docs "
            "but step 4 on the SAML config page seems outdated. Can you help?"
        ),
        "sender_email": "it-admin@mediumco.org",
        "sender_name": "IT Admin",
        "ground_truth": {
            "correct_department": Department.TECHNICAL_SUPPORT,
            "correct_urgency": UrgencyLevel.MEDIUM,
            "required_tags": {"sso", "saml", "okta", "configuration"},
            "key_response_topics": {"saml", "configuration", "steps", "guide"},
            "needs_escalation": False,
            "good_resolution_keywords": {"configured", "working", "resolved"},
        },
    },
    {
        "ticket_id": "TKT-005",
        "subject": "Data retention policy — GDPR request",
        "body": (
            "We are undergoing a GDPR audit and need your written data retention "
            "and deletion policy. Specifically: how long do you retain user logs, "
            "do you have a DPA we can sign, and what is your sub-processor list?"
        ),
        "sender_email": "dpo@eucompany.de",
        "sender_name": "Klaus Weber",
        "ground_truth": {
            "correct_department": Department.LEGAL,
            "correct_urgency": UrgencyLevel.HIGH,
            "required_tags": {"gdpr", "legal", "data-retention", "compliance"},
            "key_response_topics": {"gdpr", "dpa", "data retention", "legal team"},
            "needs_escalation": False,
            "good_resolution_keywords": {"dpa signed", "policy sent", "compliant"},
        },
    },
    # -----------------------------------------------------------------------
    # MEDIUM tickets — ambiguous signal, multi-action required
    # -----------------------------------------------------------------------
    {
        "ticket_id": "TKT-006",
        "subject": "My account is locked and I have a board demo in 2 hours",
        "body": (
            "I can't log in to my account — it says 'account suspended'. "
            "I have a critical board presentation in 2 hours where I need to "
            "show your platform live. I'm a paying Pro subscriber (since 2021). "
            "Please unlock immediately. This is extremely time-sensitive."
        ),
        "sender_email": "ceo@fastgrowth.com",
        "sender_name": "Marcus Rivera",
        "ground_truth": {
            "correct_department": Department.CUSTOMER_SUCCESS,
            "correct_urgency": UrgencyLevel.CRITICAL,
            "required_tags": {"account-locked", "urgent", "enterprise-customer"},
            "key_response_topics": {"unlock", "immediate", "apologize", "escalate"},
            "needs_escalation": True,
            "good_resolution_keywords": {"unlocked", "restored access", "resolved"},
        },
    },
    {
        "ticket_id": "TKT-007",
        "subject": "Cancellation request + refund for annual plan",
        "body": (
            "I'd like to cancel my annual subscription and get a pro-rated refund "
            "for the remaining 8 months. I'm leaving because the reporting features "
            "don't meet our needs. Invoice #9034, paid $1,200. "
            "Please confirm the cancellation and refund timeline."
        ),
        "sender_email": "finance@retailer.biz",
        "sender_name": "Patricia Lee",
        "ground_truth": {
            "correct_department": Department.BILLING,
            "correct_urgency": UrgencyLevel.MEDIUM,
            "required_tags": {"cancellation", "refund", "annual-plan", "churn-risk"},
            "key_response_topics": {"cancellation", "refund", "timeline", "confirm"},
            "needs_escalation": False,
            "good_resolution_keywords": {"cancelled", "refund issued", "confirmed"},
        },
    },
    # -----------------------------------------------------------------------
    # HARD tickets — multi-turn, escalation, full resolution required
    # -----------------------------------------------------------------------
    {
        "ticket_id": "TKT-008",
        "subject": "Data loss — all our project files are gone",
        "body": (
            "URGENT: All project files in workspace 'Acme-Q1' have disappeared. "
            "Last backup shown was 3 days ago but we've been working daily. "
            "We have a client deadline TOMORROW morning. This is catastrophic. "
            "Account: acme@enterprise.com, workspace ID: ws-39182."
        ),
        "sender_email": "acme@enterprise.com",
        "sender_name": "Acme Corp",
        "ground_truth": {
            "correct_department": Department.TECHNICAL_SUPPORT,
            "correct_urgency": UrgencyLevel.CRITICAL,
            "required_tags": {"data-loss", "critical", "enterprise", "backup"},
            "key_response_topics": {"data", "recovery", "escalate", "urgent", "backup"},
            "needs_escalation": True,
            "good_resolution_keywords": {"data restored", "files recovered", "resolved"},
            "follow_up_message": (
                "Thank you for the quick response. We found some files are back "
                "but project 'Acme-Q1-Sprint3' is still missing. Can you check again?"
            ),
            "follow_up_response_topics": {"specific project", "sprint3", "checking"},
        },
    },
    {
        "ticket_id": "TKT-009",
        "subject": "Billing discrepancy + threat of chargeback",
        "body": (
            "I've been charged $299/month for the past 6 months but my contract "
            "clearly states $199/month. Total overcharge: $600. I have the signed "
            "contract. If this isn't resolved with a full refund within 48 hours "
            "I will dispute all 6 charges with my bank. Account: robert@agency.co"
        ),
        "sender_email": "robert@agency.co",
        "sender_name": "Robert Chen",
        "ground_truth": {
            "correct_department": Department.BILLING,
            "correct_urgency": UrgencyLevel.CRITICAL,
            "required_tags": {"billing-dispute", "chargeback-risk", "contract", "refund"},
            "key_response_topics": {"contract", "overcharge", "refund", "investigate", "apologize"},
            "needs_escalation": True,
            "good_resolution_keywords": {"$600 refunded", "corrected", "resolved", "apologize"},
            "follow_up_message": (
                "I've been waiting 24 hours with no update on my refund. "
                "I'm filing the chargeback now if I don't hear back."
            ),
            "follow_up_response_topics": {"refund status", "timeline", "processing", "urgent"},
        },
    },
]

# Build a lookup dict for easy access
TICKET_LOOKUP = {t["ticket_id"]: t for t in TICKETS}


def calculate_complexity(ticket: dict) -> float:
    """
    Computes a continuous complexity score in [0.0, 1.0] for a ticket.
    Based on:
    - Number of issues/topics mentioned (derived from key topics & required tags)
    - Body text length
    - Urgency level
    - Department ambiguity
    - Multi-turn follow-up expectation
    """
    # 1. Base on word count (up to 150 words)
    body = ticket.get("body", "")
    words = len(body.split())
    size_score = min(1.0, words / 150.0)

    # 2. Key response topics & required tags density
    gt = ticket.get("ground_truth", {})
    topics_count = len(gt.get("key_response_topics", []))
    tags_count = len(gt.get("required_tags", []))
    info_density = min(1.0, (topics_count + tags_count) / 10.0)

    # 3. Urgency contribution
    urgency = gt.get("correct_urgency", "low")
    urgency_weights = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 1.0}
    urg_score = urgency_weights.get(urgency, 0.2)

    # 4. Multi-turn turn expectation
    has_follow_up = 1.0 if gt.get("follow_up_message") else 0.0

    # 5. Escalation requirement
    needs_esc = 1.0 if gt.get("needs_escalation") else 0.0

    # Combine weights:
    # 25% size, 25% info density, 15% urgency, 20% follow_up, 15% escalation
    score = (
        0.25 * size_score +
        0.25 * info_density +
        0.15 * urg_score +
        0.20 * has_follow_up +
        0.15 * needs_esc
    )
    return round(score, 4)

