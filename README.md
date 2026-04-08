---
title: NewsWriterEnv
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - news-writing
  - nlp
  - crewai
app_port: 7860
---

# NewsWriterEnv

An OpenEnv-compliant reinforcement learning environment for **news article generation**.

An RL agent receives a topic and optional writing constraints as its observation, writes a news article as its action, and receives a deterministic reward score based on article quality.

---

## Environment Description

**Real-world task:** News writing is a genuine human task — journalists write articles on assigned topics under constraints (word limits, required keywords, structured sections). This environment trains agents to produce high-quality, well-structured news content.

**Powered by CrewAI:** Each episode uses a two-agent CrewAI pipeline (researcher + writer) with live Serper web search to generate fresh, real-world content. No hardcoded articles — every episode is unique.

---

## Action & Observation Spaces

### Observation
```json
{
  "topic": "AI in healthcare",
  "difficulty": "easy",
  "constraints": {},
  "instructions": "Write a news article about: 'AI in healthcare'...",
  "step_count": 0
}
```

### Action
```json
{
  "article": "# AI is Transforming Healthcare\n\nArtificial intelligence..."
}
```

### Reward
- Type: `float` in `[0.0, 1.0]`
- Fully deterministic rule-based grader
- Partial credit for partial completion

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| Basic Article | Easy | 200+ words, 3+ paragraphs, markdown title, on-topic |
| Constrained Writing | Medium | Required keywords + word count range (250–500 words) |
| Investigative Article | Hard | Required sections, 2+ statistics, 400–700 words |

---

## Reward Function

Each task uses a weighted combination of rule-based criteria:

**Easy (weights):**
- Word count ≥ 200 → 25%
- Paragraph count ≥ 3 → 25%
- Has markdown title → 25%
- Topic relevance → 25%

**Medium adds:**
- Keywords present → 30%
- Word count in range → 30%
- Easy criteria → 40%

**Hard adds:**
- Section headings present → 25%
- Statistics/data points → 20%
- Word count in range → 15%
- Deep topic relevance → 10%
- Easy criteria → 30%

---

## Setup & Usage

### Requirements
- Python 3.10+
- Google API key (Gemini)
- Serper API key (web search)

### Install
```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

### Run the server
```bash
python server/server.py
```

### API Endpoints
```
GET  /health   → Health check
POST /reset    → Start new episode, returns observation
POST /step     → Submit article, returns reward + info
GET  /state    → Current environment state
GET  /tasks    → List all tasks
```

### Example usage
```python
import requests

# Start episode
obs = requests.post("http://localhost:7860/reset").json()
print(obs["observation"]["topic"])  # e.g. "AI in healthcare"

# Submit article
result = requests.post("http://localhost:7860/step", json={
    "article": "# AI in Healthcare\n\nArtificial intelligence is..."
}).json()
print(result["reward"])  # e.g. 0.875
```

---

## Docker

### Build
```bash
docker build -t news-writer-env -f server/Dockerfile .
```

### Run
```bash
docker run -p 7860:7860 \
  -e GOOGLE_API_KEY=your_key \
  -e SERPER_API_KEY=your_key \
  -e DIFFICULTY=easy \
  news-writer-env
```

---

## Baseline Inference

Run the baseline agent against all 3 tasks:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_openai_key
export ENV_URL=http://localhost:7860

python inference.py
```

### Baseline Scores (gpt-4o-mini)
| Task | Mean Reward |
|------|------------|
| Easy | ~0.85 |
| Medium | ~0.72 |
| Hard | ~0.61 |
| Overall | ~0.73 |

---

## HuggingFace Spaces Deployment

1. Create a new HF Space with Docker SDK
2. Push this repo to the Space
3. Add secrets: `GOOGLE_API_KEY`, `SERPER_API_KEY`
4. Tag the Space with `openenv`

The Space will auto-build and expose the API at your Space URL.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key for CrewAI agents |
| `SERPER_API_KEY` | Yes | Serper API key for web search |
| `API_BASE_URL` | Yes (inference) | LLM API endpoint |
| `MODEL_NAME` | Yes (inference) | Model identifier |
| `HF_TOKEN` | Yes (inference) | HuggingFace/API key |
| `DIFFICULTY` | No | easy/medium/hard/random (default: easy) |
| `PORT` | No | Server port (default: 7860) |
