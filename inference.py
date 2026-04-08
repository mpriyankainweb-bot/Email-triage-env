import os
from openai import OpenAI
from environment import EmailTriageEnv, Action
import json

# ── Environment variables (exact names required by hackathon) ────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("Warning: HF_TOKEN not found, using fallback mode")
    HF_TOKEN = "dummy"
# ── OpenAI client (required by hackathon) ───────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage-env"


def build_prompt(obs) -> str:
    emails_text = ""
    for e in obs.emails[:1]:
        emails_text += (
            f"\nEmail ID: {e['id']}"
            f"\nFrom: {e['from']}"
            f"\nSubject: {e['subject']}"
            f"\nBody: {e['body']}\n"
        )
    return f"""You are an email assistant. Your task:
{obs.task_description}

Current email to process:
{emails_text}
Step {obs.current_step + 1} of {obs.max_steps}. Score so far: {obs.score_so_far}

Respond with ONLY a valid JSON object matching the action format. No explanation. Just JSON.
Example for easy task: {{"action_type": "classify", "payload": {{"email_id": "e1", "label": "spam"}}}}
Example for medium task: {{"action_type": "prioritize", "payload": {{"email_id": "p1", "priority": "high"}}}}
Example for hard task: {{"action_type": "reply", "payload": {{"email_id": "c1", "reply_text": "Dear customer, I sincerely apologize..."}}}}"""


def parse_action(text: str, task_id: str) -> tuple[Action, str | None]:
    """Parse LLM response into Action. Returns (action, error_or_none)."""
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        data = json.loads(raw)
        return Action(action_type=data["action_type"], payload=data["payload"]), None
    except Exception as e:
        # Fallback actions so the episode can continue
        fallbacks = {
            "easy":   Action(action_type="classify",   payload={"email_id": "e1", "label": "ham"}),
            "medium": Action(action_type="prioritize", payload={"email_id": "p1", "priority": "medium"}),
            "hard":   Action(action_type="reply",      payload={"email_id": "c1", "reply_text": "Dear customer, I apologize for the inconvenience. We will investigate and issue a refund immediately. Please contact us so we can resolve this for you."}),
        }
        return fallbacks[task_id], str(e)


def run_task(task_id: str):
    """Run one task and print hackathon-required output format."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    done = False
    step_num = 0
    rewards = []
    last_error = None

    # ── [START] line ─────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done:
            step_num += 1
            prompt = build_prompt(obs)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1,
                )
                raw_text = response.choices[0].message.content
                action, parse_error = parse_action(raw_text, task_id)
                last_error = parse_error
            except Exception as e:
                last_error = str(e)
                # Use fallback on API error
                fallbacks = {
                    "easy":   Action(action_type="classify",   payload={"email_id": "e1", "label": "ham"}),
                    "medium": Action(action_type="prioritize", payload={"email_id": "p1", "priority": "medium"}),
                    "hard":   Action(action_type="reply",      payload={"email_id": "c1", "reply_text": "Dear customer, I apologize for the inconvenience. We will investigate and issue a refund."}),
                }
                action = fallbacks[task_id]

            obs, reward, done, info = env.step(action)
            rewards.append(reward.value)

            error_str = last_error if last_error else "null"
            action_str = json.dumps(action.dict()).replace(" ", "")

            # ── [STEP] line ───────────────────────────────────────────────────
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward.value:.2f} done={str(done).lower()} error={error_str}",
                flush=True,
            )

        success = True

    except Exception as e:
        success = False
        last_error = str(e)
        print(
            f"[STEP] step={step_num} action=null reward=0.00 done=true error={last_error}",
            flush=True,
        )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ── [END] line ────────────────────────────────────────────────────────────
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
        flush=True,
    )

    return rewards


def main():
    for task_id in TASKS:
        run_task(task_id)
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    main()
