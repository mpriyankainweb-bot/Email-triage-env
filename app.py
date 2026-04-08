"""
Gradio web UI for Email Triage Environment.
This is what runs on Hugging Face Spaces.
"""

import json
import gradio as gr
from environment import EmailTriageEnv, Action

# Global env state (one session)
envs = {}


def start_task(task_id):
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    envs[task_id] = env

    emails_text = format_emails(obs.emails)
    return (
        f"**Task started: {task_id.upper()}**\n\n{obs.task_description}",
        emails_text,
        f"Step {obs.current_step + 1}/{obs.max_steps} | Score: {obs.score_so_far}",
        "",
    )


def take_action(task_id, action_json):
    env = envs.get(task_id)
    if not env:
        return "Please start the task first!", "", "", ""

    try:
        data = json.loads(action_json)
        action = Action(action_type=data["action_type"], payload=data["payload"])
    except Exception as e:
        return f"Invalid JSON: {e}", "", "", ""

    obs, reward, done, info = env.step(action)
    emails_text = format_emails(obs.emails)

    result = f"**Reward:** {reward.value:.2f}\n**Reason:** {reward.reason}"
    if done:
        final = env.final_score()
        result += f"\n\n🏁 **Task complete! Final score: {final:.3f}**"

    status = f"Step {obs.current_step}/{obs.max_steps} | Score so far: {obs.score_so_far}"
    return result, emails_text, status, ""


def format_emails(emails):
    if not emails:
        return "No more emails."
    out = []
    for e in emails:
        out.append(
            f"**ID:** {e['id']}  \n"
            f"**From:** {e['from']}  \n"
            f"**Subject:** {e['subject']}  \n"
            f"**Body:** {e['body']}\n\n---"
        )
    return "\n".join(out)


EXAMPLE_ACTIONS = {
    "easy":   '{"action_type": "classify", "payload": {"email_id": "e1", "label": "spam"}}',
    "medium": '{"action_type": "prioritize", "payload": {"email_id": "p1", "priority": "high"}}',
    "hard":   '{"action_type": "reply", "payload": {"email_id": "c1", "reply_text": "Dear customer, I sincerely apologize for the delay with your order. We will investigate and issue a full refund immediately. Please contact us directly so we can assist you."}}',
}

with gr.Blocks(title="Email Triage Environment") as demo:
    gr.Markdown("# 📧 Email Triage Environment\nAn OpenEnv-compatible real-world task environment for AI agents.")

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Select Task",
        )
        start_btn = gr.Button("▶ Start Task", variant="primary")

    task_info = gr.Markdown("Select a task and click Start.")
    status_box = gr.Textbox(label="Status", interactive=False)
    emails_box = gr.Markdown(label="Current Emails")

    gr.Markdown("### Enter your action as JSON:")
    action_input = gr.Textbox(
        label="Action JSON",
        placeholder='{"action_type": "classify", "payload": {"email_id": "e1", "label": "spam"}}',
        lines=3,
    )
    action_btn = gr.Button("Submit Action", variant="secondary")
    reward_output = gr.Markdown(label="Result")

    def load_example(task_id):
        return EXAMPLE_ACTIONS.get(task_id, "")

    task_dropdown.change(load_example, inputs=task_dropdown, outputs=action_input)

    start_btn.click(
        start_task,
        inputs=task_dropdown,
        outputs=[task_info, emails_box, status_box, action_input],
    )
    action_btn.click(
        take_action,
        inputs=[task_dropdown, action_input],
        outputs=[reward_output, emails_box, status_box, action_input],
    )

    gr.Markdown("""
---
**Tasks:**
- 🟢 **Easy** — Spam detection (classify emails as spam/ham)
- 🟡 **Medium** — Priority assignment (high / medium / low)
- 🔴 **Hard** — Write a professional reply to a customer complaint

**OpenEnv Interface:** `reset()` → `step(action)` → `(observation, reward, done, info)`
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
