# 📧 Email Triage Environment

An **OpenEnv-compatible** real-world task environment where AI agents practice email management tasks — spam detection, priority assignment, and customer reply generation.

---

## Overview & Motivation

Email management is one of the most universal real-world tasks humans perform daily. This environment simulates three core email workflows of increasing difficulty, allowing AI agents to be evaluated on practical language understanding and decision-making — not toy benchmarks.

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Which task is running (`easy`, `medium`, `hard`) |
| `task_description` | string | Full instructions for the agent |
| `emails` | list | Remaining emails to process (id, from, subject, body) |
| `current_step` | int | How many steps taken so far |
| `max_steps` | int | Total steps in this episode |
| `score_so_far` | float | Cumulative score so far |

## Action Space

| `action_type` | Payload fields | Used in task |
|---|---|---|
| `classify` | `email_id`, `label` (spam/ham) | Easy |
| `prioritize` | `email_id`, `priority` (high/medium/low) | Medium |
| `reply` | `email_id`, `reply_text` | Hard |

---

## Tasks

### 🟢 Easy — Spam Detection
Classify 6 emails (3 spam, 3 legitimate) as `spam` or `ham`.
- **Reward:** 1.0 correct, 0.0 wrong, -0.1 invalid action
- **Max score:** 1.0

### 🟡 Medium — Email Prioritization
Assign `high`, `medium`, or `low` priority to 5 work emails.
- **Reward:** 1.0 correct, 0.5 adjacent priority, 0.0 fully wrong
- **Max score:** 1.0

### 🔴 Hard — Customer Reply Generation
Write a professional reply to an angry customer complaint email.
- **Reward:** Scored 0.0–1.0 based on keyword coverage (apologize, refund, resolve, etc.) and length
- **Max score:** 1.0

---

## Reward Function

Rewards are **intermediate** — given at every step, not just at episode end.

- Correct action: `+1.0`
- Adjacent/partial: `+0.5`
- Wrong answer: `0.0`
- Invalid action format: `-0.1` (penalizes loops/garbage output)
- Reply scored by keyword density (up to 1.0)

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt

# Run the web UI
python app.py

# Run baseline inference (requires HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token email-triage-env
```

### Python API

```python
from environment import EmailTriageEnv, Action

env = EmailTriageEnv(task_id="easy")
obs = env.reset()

action = Action(
    action_type="classify",
    payload={"email_id": "e1", "label": "spam"}
)

obs, reward, done, info = env.step(action)
print(reward.value, reward.reason)
print(env.final_score())
```

---

## Baseline Performance

Evaluated using `meta-llama/Llama-3.2-3B-Instruct` via Hugging Face Inference API:

| Task | Score |
|---|---|
| Easy (spam detection) | 0.833 |
| Medium (prioritization) | 0.700 |
| Hard (reply generation) | 0.620 |
| **Overall** | **0.718** |

---

## OpenEnv Compliance

- ✅ Typed `Observation`, `Action`, `Reward` models using Pydantic
- ✅ `step(action)` → `(observation, reward, done, info)`
- ✅ `reset()` → initial observation
- ✅ `state()` → current state dict
- ✅ `openenv.yaml` metadata file
- ✅ 3 tasks: easy → medium → hard
- ✅ Programmatic graders (deterministic, reproducible)
- ✅ Intermediate rewards at every step
- ✅ Baseline inference script with `HF_TOKEN`
- ✅ Deployable on Hugging Face Spaces
- ✅ Working Dockerfile
