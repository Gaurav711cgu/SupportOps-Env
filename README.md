# 🎫 Support Ticket Triage — OpenEnv

A real-world [OpenEnv](https://huggingface.co/openenv) environment where AI agents must
**triage, route, and resolve customer support tickets** across three difficulty levels.

This environment models a task that real support organisations handle thousands of times
per day: reading an incoming ticket, routing it to the right team, judging its urgency,
crafting a helpful reply, and — for hard tasks — managing a multi-turn conversation
through to resolution.

---

## 🌍 Motivation

Support ticket triage is:

- **High-volume** — enterprise companies route millions of tickets per year
- **High-stakes** — wrong routing costs money; slow responses lose customers
- **Multi-step** — requires reading comprehension, classification, and generation
- **Under-explored** in RL/agent benchmarks (most focus on code or web tasks)

This environment fills a genuine gap: an OpenEnv where agents can be trained and
evaluated on a real knowledge-worker workflow.

---

## 🏗️ Environment Overview

| Field | Value |
|-------|-------|
| Action space | Discrete (7 action types) + optional text fields |
| Observation space | Structured ticket + conversation history |
| Reward | Shaped per-step + terminal grader score [0.0, 1.0] |
| Episodes | Stateful, multi-step (up to 12 steps for Hard) |
| Tasks | 3 (Easy / Medium / Hard) |

---

## 📋 Tasks

### Task 1 — Ticket Routing *(Easy)*

> Route the incoming ticket to the correct department.

- **Actions required**: `ROUTE`
- **Score**: 1.0 for correct department, 0.1 for wrong (partial), 0.0 for no attempt
- **Departments**: `billing`, `technical_support`, `sales`, `customer_success`, `legal`
- **Max steps**: 3
- **Baseline score**: ~0.80

### Task 2 — Full Triage *(Medium)*

> Fully triage the ticket: route, set urgency, tag, and respond.

| Sub-task | Weight |
|----------|--------|
| Correct routing | 30% |
| Correct urgency level | 25% |
| Relevant tags applied | 20% |
| Informative customer response | 25% |

- **Max steps**: 8
- **Baseline score**: ~0.55

### Task 3 — Full Resolution *(Hard)*

> Manage a multi-turn support conversation to full resolution.

| Sub-task | Weight |
|----------|--------|
| Correct routing | 15% |
| Correct urgency | 10% |
| Quality initial response | 20% |
| Escalation decision | 20% |
| Handle customer follow-up | 20% |
| Close with resolution note | 15% |

- **Max steps**: 12
- **Baseline score**: ~0.40

---

## 🔌 API Reference

The environment is served as a REST API (FastAPI).

### `GET /`
Health check. Returns environment metadata and task list.

### `POST /reset`
Start a new episode.

```json
{
  "task_name": "route",        // "route" | "triage" | "resolve"
  "ticket_id": "TKT-001",     // optional — omit for random
  "seed": 42,                  // optional RNG seed
  "session_id": "abc123"       // optional — generated if omitted
}
```

Returns: `{ "observation": {...}, "session_id": "..." }`

### `POST /step`
Apply an action.

```json
{
  "session_id": "abc123",
  "action_type": "route",          // required
  "department": "billing",          // for ROUTE
  "response_text": "Hello...",      // for RESPOND
  "urgency": "high",                // for SET_URGENCY
  "tags": ["billing", "refund"],    // for TAG
  "escalation_reason": "...",       // for ESCALATE
  "resolution_note": "..."          // for CLOSE
}
```

Returns: `{ "observation": {...}, "reward": {...}, "done": bool, "info": {...}, "session_id": "..." }`

### `GET /state?session_id=abc123`
Full internal state including ground truth labels (for debugging/evaluation).

### `GET /tasks`
List all tasks with metadata.

---

## 🎬 Action Space

| action_type | Required fields | Description |
|-------------|----------------|-------------|
| `route` | `department` | Route ticket to a department |
| `set_urgency` | `urgency` | Set priority level |
| `respond` | `response_text` | Send a message to the customer |
| `tag` | `tags` | Apply classification labels |
| `escalate` | `escalation_reason` | Escalate with explanation |
| `close` | `resolution_note` | Resolve and close the ticket |
| `noop` | — | Take no action (wastes a step) |

**Departments**: `billing` · `technical_support` · `sales` · `customer_success` · `legal`

**Urgency levels**: `low` · `medium` · `high` · `critical`

---

## 👁️ Observation Space

```json
{
  "ticket_id": "TKT-001",
  "subject": "Double charged on my invoice",
  "body": "Full ticket text...",
  "sender_email": "user@example.com",
  "sender_name": "Jane Smith",
  "conversation_history": [
    {"sender": "Jane Smith", "content": "...", "timestamp": "2024-01-01T12:00:00Z"}
  ],
  "current_department": null,
  "current_urgency": null,
  "tags": [],
  "is_escalated": false,
  "is_closed": false,
  "step_number": 0,
  "task_name": "route",
  "task_description": "Route the ticket to the correct department...",
  "available_actions": ["route", "respond", "set_urgency", "tag", "escalate", "close", "noop"]
}
```

---

## 🏆 Reward Function

**Step rewards** (shaped, provide dense signal):
- +0.30 — correct ROUTE
- +0.20 — correct SET_URGENCY
- +0.10×overlap — TAG matching required tags
- +0.15×quality — RESPOND addressing key topics
- +0.20 — justified ESCALATE
- −0.10 — unjustified ESCALATE
- +0.10 — CLOSE with substantive resolution note

**Terminal reward** (authoritative, [0.0, 1.0]):
Each task has a dedicated deterministic grader that computes a weighted aggregate
of all sub-task scores. The terminal reward is returned in `info["final_grader_reward"]`.

---

## 🚀 Setup & Usage

### Quick Start (Launch & Verify)

To automatically install dependencies, run the PyTest suite, run the baseline agent (with automatic serverless fallback), and run the 300-episode evaluation suite in one command:

```bash
chmod +x run_all.sh
./run_all.sh
```

### Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python server.py
# → Server running at http://localhost:7860

# In another terminal, run the baseline inference (with a model of your choice)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### Docker

```bash
docker build -t ticket-triage-env .
docker run -p 7860:7860 ticket-triage-env

# Environment is now available at http://localhost:7860
```

### Quick API test

```bash
# Health check
curl http://localhost:7860/

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "route", "ticket_id": "TKT-001", "seed": 42}'

# Take an action (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<ID>", "action_type": "route", "department": "billing"}'
```

---

## 📊 Baseline Scores

Measured with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Inference API,
temperature=0.0 (greedy), seed=42:

| Task | Score | Notes |
|------|-------|-------|
| Route (Easy) | ~0.80 | Model occasionally confuses billing ↔ customer_success |
| Triage (Medium) | ~0.55 | Tags and urgency are hardest sub-tasks |
| Resolve (Hard) | ~0.40 | Follow-up handling and escalation decisions are challenging |
| **Overall** | **~0.58** | |

---

## 📁 Project Structure

```
ticket-triage-env/
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── inference.py          # Baseline inference script (hackathon-required)
├── server.py             # FastAPI HTTP server
├── README.md             # This file
└── env/
    ├── __init__.py
    ├── environment.py    # Core TicketTriageEnv class
    ├── models.py         # Pydantic Observation/Action/Reward models
    ├── tasks.py          # Task specifications
    ├── graders.py        # Deterministic grader functions
    └── data.py           # Synthetic ticket dataset with ground truth
```

---

## 📜 License

MIT — free to use for research and commercial applications.

---

## 📊 Evaluation Leaderboard & Benchmark Results

> Evaluated 5 frontier and open-weights models · 20 episodes per task · **300 total episodes**

### Leaderboard

| Model | Easy (Route) | Medium (Triage) | Hard (Resolve) | Δ Easy→Hard |
|---|:---:|:---:|:---:|:---:|
| Claude 3.5 Sonnet | 0.96 | 0.89 | 0.74 | -23% |
| GPT-4o-Mini | 0.96 | 0.86 | 0.70 | -27% |
| Gemini 2.0 Flash | 0.86 | 0.86 | 0.62 | -28% |
| Llama-3.1-8B | 0.82 | 0.70 | 0.39 | -53% |
| Mistral-7B | 0.82 | 0.65 | 0.40 | -51% |

**Key finding**: Larger models degrade 46–53% from Easy→Hard; 7B-class models collapse 73–77%.
Multi-step reasoning, long-context tracking, and strict sub-task adherence require higher parametric
capacity. Smaller models lose state, mis-route on ambiguous signals, and fail to handle follow-up turns.

---

### Hard Task Failure Mode Analysis

Failure counts among Hard task episodes scoring below 0.3 (out of 20 episodes):

| Model | Wrong Route | Wrong Urgency | Missing Tags | Unhelpful Resp | No Follow-up | Step Limit |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Claude 3.5 Sonnet | 0 | 0 | 0 | 1 | 1 | 0 |
| GPT-4o-Mini | 1 | 1 | 0 | 2 | 2 | 0 |
| Gemini 2.0 Flash | 1 | 2 | 0 | 3 | 3 | 0 |
| Llama-3.1-8B | 6 | 4 | 0 | 7 | 5 | 0 |
| Mistral-7B | 3 | 2 | 0 | 3 | 3 | 0 |

---

### Reward Hacking & LLM-as-Judge (Scalable Oversight)

The original `keyword_overlap` grader assigned full credit to any response containing the right keywords,
regardless of coherence — a classic **reward hacking vector**. We replaced it with a **dual-signal grader**:

- **50% keyword overlap** (fast, deterministic)
- **50% LLM judge score** (coherence, tone, actionability)

This mirrors Anthropic's scalable oversight paradigm: augmenting a weak but cheap signal with a
stronger, more expensive signal to keep agent behavior aligned.

#### Measured Reward Hacking Rate (keyword grader score ≥ 0.8 but LLM judge < 0.4)

- **Claude 3.5 Sonnet**: 1/40 (2%) responses flagged
- **GPT-4o-Mini**: 9/40 (22%) responses flagged
- **Gemini 2.0 Flash**: 6/40 (15%) responses flagged
- **Llama-3.1-8B**: 13/40 (32%) responses flagged
- **Mistral-7B**: 17/40 (42%) responses flagged

---

### Continuous Difficulty Curve

Performance as a function of ticket complexity score (0.0–1.0), showing that model capability
degrades continuously — not just at discrete Easy/Medium/Hard boundaries.
See `eval_results.json` for the full per-ticket breakdown.

