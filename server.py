"""
OpenEnv-compliant WebSocket + HTTP server for NewsWriterEnv.
Exposes: POST /reset, POST /step, GET /state, GET /health
"""

import os
import sys
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env import NewsWriterEnv, Action, Observation
from crew import run_news_crew

app = FastAPI(
    title="NewsWriterEnv",
    description="OpenEnv environment for RL-based news article generation",
    version="1.0.0",
)

# Env instances per difficulty (one per level + random)
_envs = {
    "easy":   NewsWriterEnv(difficulty="easy"),
    "medium": NewsWriterEnv(difficulty="medium"),
    "hard":   NewsWriterEnv(difficulty="hard"),
    "random": NewsWriterEnv(difficulty="random"),
}
_active_difficulty = os.getenv("DIFFICULTY", "easy")
env = _envs[_active_difficulty]


class StepRequest(BaseModel):
    article: str


class ResetResponse(BaseModel):
    observation: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


@app.get("/health")
def health():
    return {"status": "ok", "env": "NewsWriterEnv", "version": "1.0.0"}


@app.get("/")
def serve_ui():
    """Serve the dashboard UI."""
    return FileResponse("index.html")


@app.post("/demo/run")
def demo_run(difficulty: str = "easy"):
    """
    End-to-end demo: resets env, generates an article via CrewAI, and steps the env.
    Used exclusively by the UI dashboard.
    """
    if difficulty not in _envs:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
        
    target_env = _envs[difficulty]
    obs = target_env.reset()
    
    # Create the article using CrewAI agents
    try:
        article = run_news_crew(obs.topic, obs.constraints)
    except Exception as e:
        article = f"# Generated Article Error\n\nSomething went wrong: {e}"
        
    # Submit article to environment to get graded
    final_obs, reward, done, info = target_env.step(Action(article=article))
    
    return {
        "topic": obs.topic,
        "instructions": obs.instructions,
        "article": article,
        "reward": reward,
        "breakdown": info.get("reward_breakdown", {})
    }


@app.post("/reset", response_model=ResetResponse)
def reset(difficulty: str = None):
    """
    Start a new episode.
    Optional query param: difficulty=easy|medium|hard|random
    Returns initial observation with topic and instructions.
    """
    global env, _active_difficulty

    if difficulty:
        if difficulty not in _envs:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=f"difficulty must be one of: {list(_envs.keys())}"
            )
        _active_difficulty = difficulty
        env = _envs[difficulty]

    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit an article and receive a reward.
    Body: { "article": "your article text here" }
    """
    if not request.article or not request.article.strip():
        raise HTTPException(status_code=400, detail="Article cannot be empty")

    action = Action(article=request.article)
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return current environment state."""
    return env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Basic Article Generation",
                "description": (
                    "Write a coherent news article on a well-known topic. "
                    "Requires: 200+ words, 3+ paragraphs, markdown title, on-topic content."
                ),
                "difficulty": "easy",
                "reward_range": [0.001, 0.999],
            },
            {
                "id": "medium",
                "name": "Constrained Article Writing",
                "description": (
                    "Write an article meeting specific constraints: "
                    "required keywords, word count range, and proper structure."
                ),
                "difficulty": "medium",
                "reward_range": [0.001, 0.999],
            },
            {
                "id": "hard",
                "name": "Structured Investigative Article",
                "description": (
                    "Write a full investigative piece with required sections "
                    "(Introduction, Current Trends, Challenges, Future Outlook), "
                    "2+ statistics, and balanced tone on a niche/controversial topic."
                ),
                "difficulty": "hard",
                "reward_range": [0.001, 0.999],
            },
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
