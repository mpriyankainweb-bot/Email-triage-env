"""
Email Triage Environment - OpenEnv Compatible
A real-world task simulation for AI agents to practice email management.
"""

import random
from typing import Any
from pydantic import BaseModel


# ── Typed models (OpenEnv requirement) ──────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_description: str
    emails: list[dict]
    current_step: int
    max_steps: int
    score_so_far: float


class Action(BaseModel):
    action_type: str   # "classify", "prioritize", or "reply"
    payload: dict      # task-specific data


class Reward(BaseModel):
    value: float
    reason: str
    done: bool


# ── Email data ───────────────────────────────────────────────────────────────

SPAM_EMAILS = [
    {"id": "e1", "subject": "You WON a prize!!!", "from": "noreply@scam.xyz",
     "body": "Click here to claim $1000 gift card NOW!!!"},
    {"id": "e2", "subject": "Cheap meds online", "from": "pharma@bulk.ru",
     "body": "Buy prescription drugs without a prescription!"},
    {"id": "e3", "subject": "Hot singles in your area", "from": "dating@spam.com",
     "body": "Meet people near you tonight!"},
]

HAM_EMAILS = [
    {"id": "e4", "subject": "Team meeting tomorrow", "from": "boss@company.com",
     "body": "Hi, we have a standup at 10am tomorrow. Please be on time."},
    {"id": "e5", "subject": "Your invoice #4521", "from": "billing@service.com",
     "body": "Your monthly invoice is ready. Amount due: $29.99"},
    {"id": "e6", "subject": "Project update needed", "from": "client@partner.org",
     "body": "Could you send us the latest status on the project?"},
]

PRIORITY_EMAILS = [
    {"id": "p1", "subject": "URGENT: Server is down!", "from": "ops@company.com",
     "body": "Production server crashed. Customers cannot access the app.", "correct_priority": "high"},
    {"id": "p2", "subject": "Lunch menu this week", "from": "hr@company.com",
     "body": "Check out what's on the cafeteria menu this week.", "correct_priority": "low"},
    {"id": "p3", "subject": "Q3 report due Friday", "from": "manager@company.com",
     "body": "Reminder: quarterly report is due end of week.", "correct_priority": "medium"},
    {"id": "p4", "subject": "Security breach detected", "from": "security@company.com",
     "body": "Unauthorized login attempt from unknown IP. Action needed.", "correct_priority": "high"},
    {"id": "p5", "subject": "Birthday party invite", "from": "friend@personal.com",
     "body": "Come to my birthday party this Saturday!", "correct_priority": "low"},
]

COMPLAINT_EMAIL = {
    "id": "c1",
    "subject": "Very disappointed with your service",
    "from": "angry.customer@email.com",
    "body": (
        "I have been waiting 2 weeks for my order and it still hasn't arrived. "
        "Your customer support is not responding. I want a refund immediately. "
        "This is completely unacceptable."
    ),
}

GOOD_REPLY_KEYWORDS = [
    "apologize", "sorry", "refund", "resolve", "investigate",
    "understand", "assist", "help", "order", "contact"
]


# ── Tasks ────────────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "id": "easy",
        "description": (
            "EASY TASK — Spam Detection:\n"
            "You will receive 6 emails one by one. "
            "For each email, classify it as 'spam' or 'ham' (not spam).\n"
            "Action format: {\"action_type\": \"classify\", \"payload\": {\"email_id\": \"e1\", \"label\": \"spam\"}}"
        ),
        "emails": SPAM_EMAILS + HAM_EMAILS,
        "answers": {e["id"]: "spam" for e in SPAM_EMAILS} | {e["id"]: "ham" for e in HAM_EMAILS},
    },
    "medium": {
        "id": "medium",
        "description": (
            "MEDIUM TASK — Email Prioritization:\n"
            "You will see 5 work emails. Assign each a priority: 'high', 'medium', or 'low'.\n"
            "Action format: {\"action_type\": \"prioritize\", \"payload\": {\"email_id\": \"p1\", \"priority\": \"high\"}}"
        ),
        "emails": PRIORITY_EMAILS,
        "answers": {e["id"]: e["correct_priority"] for e in PRIORITY_EMAILS},
    },
    "hard": {
        "id": "hard",
        "description": (
            "HARD TASK — Reply to a customer complaint:\n"
            "Read the angry customer email below and write a professional reply.\n"
            "Action format: {\"action_type\": \"reply\", \"payload\": {\"email_id\": \"c1\", \"reply_text\": \"Dear customer...\"}}"
        ),
        "emails": [COMPLAINT_EMAIL],
        "answers": {},
    },
}


# ── Environment class ────────────────────────────────────────────────────────

class EmailTriageEnv:
    """OpenEnv-compatible Email Triage Environment."""

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASKS, f"task_id must be one of {list(TASKS.keys())}"
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._step = 0
        self._score = 0.0
        self._done = False
        self._emails = list(self.task["emails"])
        self._email_index = 0
        self._max_steps = len(self._emails)
        self._history: list[dict] = []

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self._step = 0
        self._score = 0.0
        self._done = False
        self._email_index = 0
        self._history = []
        random.shuffle(self._emails)
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Take one action and return (observation, reward, done, info)."""
        if self._done:
            reward = Reward(value=0.0, reason="Episode already finished.", done=True)
            return self._make_observation(), reward, True, {}

        reward = self._grade_action(action)
        self._score += reward.value
        self._step += 1
        self._email_index += 1
        self._history.append({"step": self._step, "action": action.dict(), "reward": reward.dict()})

        if self._email_index >= self._max_steps:
            self._done = True
            reward.done = True

        return self._make_observation(), reward, self._done, {"history": self._history}

    def state(self) -> dict:
        """Return full current state."""
        return {
            "task_id": self.task_id,
            "step": self._step,
            "score": self._score,
            "done": self._done,
            "history": self._history,
        }

    # ── Graders ─────────────────────────────────────────────────────────────

    def _grade_action(self, action: Action) -> Reward:
        # Penalize wrong action type
        if action.action_type not in ("classify", "prioritize", "reply"):
            return Reward(value=-0.1, reason="Invalid action_type.", done=False)

        if self.task_id == "easy":
            return self._grade_classify(action)
        elif self.task_id == "medium":
            return self._grade_prioritize(action)
        else:
            return self._grade_reply(action)

    def _grade_classify(self, action: Action) -> Reward:
        email_id = action.payload.get("email_id", "")
        label = action.payload.get("label", "").lower()
        correct = self.task["answers"].get(email_id)

        if correct is None:
            return Reward(value=-0.1, reason=f"Unknown email_id: {email_id}", done=False)
        if label not in ("spam", "ham"):
            return Reward(value=-0.1, reason="Label must be 'spam' or 'ham'.", done=False)
        if label == correct:
            return Reward(value=1.0, reason=f"Correct! {email_id} is {correct}.", done=False)
        return Reward(value=0.0, reason=f"Wrong. {email_id} is {correct}, not {label}.", done=False)

    def _grade_prioritize(self, action: Action) -> Reward:
        email_id = action.payload.get("email_id", "")
        priority = action.payload.get("priority", "").lower()
        correct = self.task["answers"].get(email_id)

        if correct is None:
            return Reward(value=-0.1, reason=f"Unknown email_id: {email_id}", done=False)
        if priority not in ("high", "medium", "low"):
            return Reward(value=-0.1, reason="Priority must be 'high', 'medium', or 'low'.", done=False)
        if priority == correct:
            return Reward(value=1.0, reason=f"Correct! {email_id} is {correct} priority.", done=False)
        # Partial credit for adjacent priority
        adjacent = {"high": "medium", "medium": ["high", "low"], "low": "medium"}
        adj = adjacent[correct]
        if isinstance(adj, list) and priority in adj:
            return Reward(value=0.5, reason=f"Close. {email_id} is {correct}, not {priority}.", done=False)
        if priority == adj:
            return Reward(value=0.5, reason=f"Close. {email_id} is {correct}, not {priority}.", done=False)
        return Reward(value=0.0, reason=f"Wrong. {email_id} is {correct} priority.", done=False)

    def _grade_reply(self, action: Action) -> Reward:
        reply = action.payload.get("reply_text", "")
        if len(reply) < 30:
            return Reward(value=0.1, reason="Reply too short. Write at least a few sentences.", done=False)

        reply_lower = reply.lower()
        hits = sum(1 for kw in GOOD_REPLY_KEYWORDS if kw in reply_lower)
        score = min(hits / 5.0, 1.0)   # max out at 5 keywords

        # Bonus: professional greeting
        if reply_lower.startswith("dear"):
            score = min(score + 0.1, 1.0)

        reason = f"Reply scored {score:.2f}. Keywords matched: {hits}/{len(GOOD_REPLY_KEYWORDS)}."
        return Reward(value=round(score, 2), reason=reason, done=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        remaining = self._emails[self._email_index:] if self._email_index < len(self._emails) else []
        return Observation(
            task_id=self.task_id,
            task_description=self.task["description"],
            emails=remaining,
            current_step=self._step,
            max_steps=self._max_steps,
            score_so_far=round(self._score, 3),
        )

    def final_score(self) -> float:
        """Return normalized final score between 0.0 and 1.0."""
        if self._max_steps == 0:
            return 0.0
        return round(self._score / self._max_steps, 3)
