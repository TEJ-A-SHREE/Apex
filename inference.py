"""
Baseline inference script for NewsWriterEnv.
Uses OpenAI client (as required by the problem statement).

Environment variables required:
  - API_BASE_URL:      The LLM API endpoint
  - MODEL_NAME:        The model identifier
  - HF_TOKEN:          HuggingFace / API key (or OPENAI_API_KEY)
  - ENV_URL:           The running NewsWriterEnv URL (default: http://localhost:7860)
  - EPISODES_PER_TASK: Number of episodes per task (default: 3)

Usage:
  python inference.py

Stdout format (strictly followed):
  [START] {"task": "easy", "model": "gpt-4o-mini"}
  [STEP]  {"task": "easy", "episode": 1, "topic": "...", "reward": 0.85, "done": true, ...}
  [END]   {"task": "easy", "mean_reward": 0.85, "episodes": 3, "rewards": [...]}
"""

import os
import sys
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "3"))

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

DIFFICULTIES = ["easy", "medium", "hard"]


# ─── Env helpers ─────────────────────────────────────────────────────────────

def reset_env(difficulty: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/reset",
        params={"difficulty": difficulty},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(article: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"article": article},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Agent ───────────────────────────────────────────────────────────────────

def generate_article(topic: str, instructions: str) -> str:
    """Use the OpenAI-compatible client to write a news article."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional news writer. "
                    "Write clear, engaging, well-structured news articles in markdown. "
                    "Always start with a # Title heading."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Write a news article following these instructions exactly:\n\n"
                    f"{instructions}\n\nTopic: {topic}"
                ),
            },
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# ─── Task runner ─────────────────────────────────────────────────────────────

def run_task(difficulty: str) -> dict:
    """Run EPISODES_PER_TASK episodes for one difficulty level."""

    # Strictly required [START] log
    start_payload = {"task": difficulty, "model": MODEL_NAME}
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    rewards = []

    for episode in range(1, EPISODES_PER_TASK + 1):
        # Reset
        reset_resp = reset_env(difficulty)
        obs = reset_resp["observation"]
        topic = obs["topic"]
        instructions = obs["instructions"]
        constraints = obs.get("constraints", {})

        # Generate article
        try:
            article = generate_article(topic, instructions)
        except Exception as e:
            article = f"# {topic}\n\nUnable to generate article: {e}"

        # Step
        step_resp = step_env(article)
        reward = step_resp["reward"]
        done = step_resp["done"]
        info = step_resp.get("info", {})

        rewards.append(reward)

        # Strictly required [STEP] log
        step_payload = {
            "task": difficulty,
            "episode": episode,
            "topic": topic,
            "reward": reward,
            "done": done,
            "word_count": info.get("article_word_count", len(article.split())),
            "breakdown": info.get("reward_breakdown", {}),
            "constraints": constraints,
        }
        print(f"[STEP]  {json.dumps(step_payload)}", flush=True)

    mean_reward = round(sum(rewards) / len(rewards), 4)

    # Strictly required [END] log
    end_payload = {
        "task": difficulty,
        "mean_reward": mean_reward,
        "episodes": EPISODES_PER_TASK,
        "rewards": rewards,
        "model": MODEL_NAME,
    }
    print(f"[END]   {json.dumps(end_payload)}", flush=True)

    return end_payload


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== NewsWriterEnv Baseline Inference ===", flush=True)
    print(f"Model:            {MODEL_NAME}", flush=True)
    print(f"API Base URL:     {API_BASE_URL}", flush=True)
    print(f"Env URL:          {ENV_URL}", flush=True)
    print(f"Episodes/task:    {EPISODES_PER_TASK}\n", flush=True)

    # Verify env is reachable
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10).json()
        print(f"Env health: {health}\n", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot reach env at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    results = {}
    for difficulty in DIFFICULTIES:
        result = run_task(difficulty)
        results[difficulty] = result
        print(flush=True)

    # Final summary
    print("=== SUMMARY ===", flush=True)
    for diff, res in results.items():
        print(f"  {diff:8s}: mean_reward = {res['mean_reward']}", flush=True)

    overall = round(
        sum(r["mean_reward"] for r in results.values()) / len(results), 4
    )
    print(f"  {'overall':8s}: mean_reward = {overall}", flush=True)
    print(
        f"\n[SUMMARY] {json.dumps({'results': results, 'overall_mean_reward': overall, 'model': MODEL_NAME})}",
        flush=True,
    )


if __name__ == "__main__":
    main()
