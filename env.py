"""
NewsWriterEnv - An OpenEnv environment for news article generation.

An RL agent receives a topic (and optional constraints) as an observation,
submits a written article as an action, and receives a reward based on
article quality scored by deterministic rule-based graders.
"""

import random
from typing import Any
from pydantic import BaseModel, Field
from topics import sample_easy, sample_medium, sample_hard
from graders import grade
from crew import run_news_crew


# ─── Pydantic Models ────────────────────────────────────────────────────────

class Observation(BaseModel):
    topic: str = Field(description="The news topic the agent must write about")
    difficulty: str = Field(description="Task difficulty: easy | medium | hard")
    constraints: dict = Field(
        default_factory=dict,
        description=(
            "Optional writing constraints: keywords, min_words, max_words, "
            "sections, require_stats"
        ),
    )
    instructions: str = Field(
        description="Human-readable instructions for the agent"
    )
    step_count: int = Field(default=0, description="Number of steps taken in episode")


class Action(BaseModel):
    article: str = Field(
        description="The news article written by the agent in markdown format"
    )


class Reward(BaseModel):
    score: float = Field(description="Reward score between 0.0 and 1.0")
    breakdown: dict = Field(
        default_factory=dict,
        description="Partial score breakdown for each criterion"
    )


# ─── Environment ────────────────────────────────────────────────────────────

class NewsWriterEnv:
    """
    OpenEnv-compliant news writing environment.

    Episode flow:
      1. reset() → samples a topic and difficulty, returns Observation
      2. step(action) → agent submits article, receives reward + new observation
      3. Done after 1 step (single-turn episode per article)

    Tasks:
      - easy:   Write a basic article on a well-known topic
      - medium: Write with keyword and length constraints
      - hard:   Write a structured investigative piece with stats and sections
    """

    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, difficulty: str = "easy", use_crew: bool = True):
        """
        Args:
            difficulty: One of 'easy', 'medium', 'hard'. 
                        If 'random', samples a difficulty each episode.
            use_crew: If True, uses CrewAI to generate a baseline article 
                      in info. Set False for faster testing.
        """
        assert difficulty in self.DIFFICULTIES + ["random"], (
            f"difficulty must be one of {self.DIFFICULTIES + ['random']}"
        )
        self.difficulty = difficulty
        self.use_crew = use_crew
        self._current_topic = None
        self._current_constraints = None
        self._current_difficulty = None
        self._step_count = 0
        self._done = False

    def reset(self) -> Observation:
        """Start a new episode. Samples a fresh topic and difficulty."""
        self._done = False
        self._step_count = 0

        # Sample difficulty
        if self.difficulty == "random":
            self._current_difficulty = random.choice(self.DIFFICULTIES)
        else:
            self._current_difficulty = self.difficulty

        # Sample topic + constraints for this difficulty
        if self._current_difficulty == "easy":
            sample = sample_easy()
        elif self._current_difficulty == "medium":
            sample = sample_medium()
        else:
            sample = sample_hard()

        self._current_topic = sample["topic"]
        self._current_constraints = sample["constraints"]

        return Observation(
            topic=self._current_topic,
            difficulty=self._current_difficulty,
            constraints=self._current_constraints,
            instructions=self._build_instructions(),
            step_count=0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Submit an article and receive a reward.

        Args:
            action: Action containing the written article

        Returns:
            observation: Current state observation
            reward: Float score 0.0–1.0
            done: True (single-turn episodes)
            info: Dict with score breakdown and metadata
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._done = True

        # Grade the article
        score = grade(
            article=action.article,
            topic=self._current_topic,
            difficulty=self._current_difficulty,
            constraints=self._current_constraints,
        )

        # Build score breakdown for transparency
        breakdown = self._build_breakdown(action.article)

        reward = Reward(score=score, breakdown=breakdown)

        obs = Observation(
            topic=self._current_topic,
            difficulty=self._current_difficulty,
            constraints=self._current_constraints,
            instructions=self._build_instructions(),
            step_count=self._step_count,
        )

        info = {
            "reward_breakdown": breakdown,
            "topic": self._current_topic,
            "difficulty": self._current_difficulty,
            "constraints": self._current_constraints,
            "article_word_count": len(action.article.split()),
        }

        return obs, reward.score, self._done, info

    def state(self) -> dict:
        """Return current environment state."""
        return {
            "topic": self._current_topic,
            "difficulty": self._current_difficulty,
            "constraints": self._current_constraints,
            "step_count": self._step_count,
            "done": self._done,
        }

    def _build_instructions(self) -> str:
        d = self._current_difficulty
        c = self._current_constraints

        base = (
            f"Write a news article about: '{self._current_topic}'.\n"
            f"Difficulty: {d.upper()}\n\n"
        )

        if d == "easy":
            base += (
                "Requirements:\n"
                "- At least 200 words\n"
                "- At least 3 paragraphs\n"
                "- Include a markdown title (# Title)\n"
                "- Stay on topic\n"
            )
            if "starting_phrase" in c:
                base += f"- The first body paragraph MUST begin exactly with: '{c['starting_phrase']}'\n"
        elif d == "medium":
            kw = ", ".join(c.get("keywords", []))
            base += (
                f"Requirements:\n"
                f"- Include these keywords: {kw}\n"
                f"- Between {c.get('min_words', 300)} and {c.get('max_words', 500)} words\n"
            )
            if "exact_paragraphs" in c:
                base += f"- Exactly {c['exact_paragraphs']} paragraphs\n"
            else:
                base += f"- At least 3 paragraphs\n"
            base += (
                f"- Include exactly {c.get('required_quotes', 0)} blockquotes (lines starting with >)\n"
                f"- Include a markdown title\n"
                f"- Stay on topic\n"
            )
        elif d == "hard":
            sections = ", ".join(c.get("sections", []))
            base += (
                f"Requirements:\n"
                f"- Must contain these sections as ## headings: {sections}\n"
                f"- Include at least 2 specific statistics or data points\n"
                f"- Between {c.get('min_words', 400)} and {c.get('max_words', 700)} words\n"
                f"- Include exactly {c.get('exact_citations', 0)} inline citations in the exact format [Ref: X] where X is a number\n"
                f"- The main title (# headline) must be exactly {c.get('title_length', 0)} words long\n"
            )
            if "exact_word_frequency" in c:
                for w, freq in c["exact_word_frequency"].items():
                    base += f"- The exact word '{w}' MUST appear exactly {freq} times in the body\n"
            if "exact_nth_word" in c:
                nth = c["exact_nth_word"]
                base += f"- The {nth['n']}th word in the article (counting all words including headings) MUST be the word '{nth['word']}'\n"
            base += (
                f"- Balanced, investigative tone\n"
            )

        return base

    def _build_breakdown(self, article: str) -> dict:
        """Build a human-readable score breakdown."""
        from graders import (
            _count_words, _count_paragraphs, _has_title,
            _topic_relevance, _keywords_present,
            _word_count_in_range, _has_sections, _has_stats,
            _exact_quotes, _exact_citations, _title_length,
            _starts_with, _exact_word_frequency, _exact_nth_word
        )
        c = self._current_constraints
        breakdown = {
            "word_count": _count_words(article),
            "paragraph_count": _count_paragraphs(article),
            "has_title": _has_title(article),
            "topic_relevance": round(_topic_relevance(article, self._current_topic), 3),
            "keywords_present": round(
                _keywords_present(article, c.get("keywords", [])), 3
            ),
            "in_word_range": _word_count_in_range(
                article,
                c.get("min_words", 200),
                c.get("max_words", 99999),
            ),
            "sections_present": round(
                _has_sections(article, c.get("sections", [])), 3
            ),
            "has_stats": _has_stats(article),
            "exact_quotes": _exact_quotes(article, c.get("required_quotes", 0)) if "required_quotes" in c else 1.0,
            "exact_citations": _exact_citations(article, c.get("exact_citations", 3)) if "exact_citations" in c else 1.0,
            "title_length_match": _title_length(article, c.get("title_length", 6)) if "title_length" in c else 1.0,
        }
        if "starting_phrase" in c:
            breakdown["starts_with"] = _starts_with(article, c["starting_phrase"])
        if "exact_paragraphs" in c:
            breakdown["exact_paragraphs"] = 1.0 if breakdown["paragraph_count"] == c["exact_paragraphs"] else 0.0
        if "exact_word_frequency" in c:
            for w, freq in c["exact_word_frequency"].items():
                breakdown[f"freq_{w}"] = _exact_word_frequency(article, w, freq)
        if "exact_nth_word" in c:
            nth = c["exact_nth_word"]
            breakdown["nth_word_match"] = _exact_nth_word(article, nth["word"], nth["n"])
        return breakdown
