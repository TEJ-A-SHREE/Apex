"""
Local validation tests for NewsWriterEnv.
Tests graders, env flow, and API spec compliance.
Run with: python test_env.py
No API keys required — tests the graders and env logic only.
"""

import sys
import json

# ─── Test graders directly ───────────────────────────────────────────────────

def test_graders():
    from graders import grade_easy, grade_medium, grade_hard

    print("Testing graders...")

    # Good easy article
    good_article = """# AI is Transforming Healthcare

Artificial intelligence is revolutionizing the healthcare industry in profound ways.
From diagnostic tools to drug discovery, AI is helping doctors and researchers
make faster and more accurate decisions than ever before.

One of the most significant applications is in medical imaging. AI models can now
detect cancers and other diseases with accuracy that rivals or exceeds human experts.
This technology is saving lives and reducing diagnostic errors worldwide.

Looking ahead, the integration of AI in healthcare will only deepen. With growing
datasets and improving algorithms, we can expect AI to play a central role in
personalized medicine, predictive health monitoring, and drug development.
"""

    score = grade_easy(good_article, "AI in healthcare")
    assert score > 0.7, f"Good article should score > 0.7, got {score}"
    print(f"  ✓ Easy good article: {score}")

    # Bad easy article (too short, no title)
    bad_article = "AI is cool."
    score = grade_easy(bad_article, "AI in healthcare")
    assert score < 0.4, f"Bad article should score < 0.4, got {score}"
    print(f"  ✓ Easy bad article: {score}")

    # Empty article
    score = grade_easy("", "AI in healthcare")
    assert score == 0.05, f"Empty article should score 0.05, got {score}"
    print(f"  ✓ Easy empty article: {score}")

    # Medium - with keywords
    constraints = {"keywords": ["innovation", "impact", "future"], "min_words": 300, "max_words": 500}
    medium_article = good_article * 2  # make it longer
    medium_article += "\nThe innovation of AI has a massive impact on the future of medicine."
    score = grade_medium(medium_article, "AI in healthcare", constraints)
    assert 0.0 <= score <= 1.0, f"Medium score out of range: {score}"
    print(f"  ✓ Medium article with keywords: {score}")

    # Hard - with sections and stats
    hard_article = """# The Future of AI in Healthcare

## Introduction
Artificial intelligence is fundamentally changing how healthcare is delivered.

## Current Trends
The global AI healthcare market is projected to reach $45.2 billion by 2026,
growing at 44.9% annually. Over 86% of healthcare providers are now using AI tools.

## Challenges
Despite progress, algorithmic bias remains a significant concern. Studies show
AI models trained on non-diverse datasets can underperform for certain populations.

## Future Outlook
With continued investment and regulation, AI will become a standard part of clinical
workflows. The next decade will see AI move from pilot programs to core infrastructure.
"""
    hard_constraints = {
        "sections": ["Introduction", "Current Trends", "Challenges", "Future Outlook"],
        "require_stats": True,
        "min_words": 400,
        "max_words": 700,
    }
    score = grade_hard(hard_article, "AI in healthcare", hard_constraints)
    assert 0.0 <= score <= 1.0, f"Hard score out of range: {score}"
    print(f"  ✓ Hard structured article: {score}")

    print("  ✓ All grader tests passed!\n")


# ─── Test env flow ────────────────────────────────────────────────────────────

def test_env_flow():
    from env import NewsWriterEnv, Action

    print("Testing environment flow...")

    for difficulty in ["easy", "medium", "hard"]:
        env = NewsWriterEnv(difficulty=difficulty)

        # reset() returns valid observation
        obs = env.reset()
        assert obs.topic, "Topic should not be empty"
        assert obs.difficulty == difficulty
        assert obs.instructions, "Instructions should not be empty"
        assert obs.step_count == 0
        print(f"  ✓ reset() works for {difficulty}: topic='{obs.topic}'")

        # state() works
        state = env.state()
        assert state["topic"] == obs.topic
        assert not state["done"]
        print(f"  ✓ state() works for {difficulty}")

        # step() returns valid reward
        test_article = f"""# Test Article on {obs.topic}

This is a test article about {obs.topic}. It contains multiple paragraphs
to ensure the grader can evaluate it properly. The topic is very important
in today's world of technology and innovation.

The impact of {obs.topic} is growing every year with new developments emerging.
Researchers and industry experts are paying close attention to these trends.

Looking forward, {obs.topic} will continue to shape our future in meaningful ways.
The technology brings both opportunities and challenges that society must address.
"""
        action = Action(article=test_article)
        new_obs, reward, done, info = env.step(action)

        assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
        assert done is True, "Episode should be done after 1 step"
        assert "reward_breakdown" in info
        print(f"  ✓ step() works for {difficulty}: reward={reward}")

        # Can't step again after done
        try:
            env.step(action)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            print(f"  ✓ Double-step correctly raises RuntimeError for {difficulty}")

        # reset() starts fresh
        obs2 = env.reset()
        assert env.state()["done"] is False
        print(f"  ✓ reset() after done works for {difficulty}\n")

    print("  ✓ All env flow tests passed!\n")


# ─── Test determinism ─────────────────────────────────────────────────────────

def test_determinism():
    from graders import grade

    print("Testing grader determinism...")

    article = """# Quantum Computing in Finance

Quantum computing is set to revolutionize the financial sector with unprecedented speed.
Banks and hedge funds are investing billions into this transformative technology.
The innovation will have a significant impact on the future of trading and risk management.

The technology enables calculations that would take classical computers thousands of years.
Financial institutions are exploring quantum algorithms for portfolio optimization,
fraud detection, and derivatives pricing with remarkable early results.

Major players including IBM, Google, and specialized fintech firms are racing to
develop quantum-ready solutions for the industry.
"""

    constraints = {"keywords": ["innovation", "impact", "future"], "min_words": 250, "max_words": 500}

    # Same input → same score every time
    score1 = grade(article, "quantum computing in finance", "medium", constraints)
    score2 = grade(article, "quantum computing in finance", "medium", constraints)
    score3 = grade(article, "quantum computing in finance", "medium", constraints)

    assert score1 == score2 == score3, f"Grader not deterministic: {score1}, {score2}, {score3}"
    print(f"  ✓ Same article always scores {score1} (deterministic)")

    # Different articles → different scores
    short_article = "Quantum computing is interesting."
    score_short = grade(short_article, "quantum computing in finance", "medium", constraints)
    assert score_short != score1, "Different articles should score differently"
    print(f"  ✓ Short article scores {score_short} (different from {score1})")

    print("  ✓ All determinism tests passed!\n")


# ─── Test topics ──────────────────────────────────────────────────────────────

def test_topics():
    from topics import sample_easy, sample_medium, sample_hard

    print("Testing topic sampling...")

    # Each call returns a valid dict
    for fn, name in [(sample_easy, "easy"), (sample_medium, "medium"), (sample_hard, "hard")]:
        sample = fn()
        assert "topic" in sample and sample["topic"], f"{name} topic empty"
        assert "constraints" in sample, f"{name} missing constraints"
        print(f"  ✓ {name}: topic='{sample['topic']}'")

    # Sampling produces varied topics
    topics = {sample_easy()["topic"] for _ in range(20)}
    assert len(topics) > 1, "Easy topics should vary"
    print(f"  ✓ Topic variety: {len(topics)} unique topics in 20 samples")

    print("  ✓ All topic tests passed!\n")


# ─── Run all ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("NewsWriterEnv — Local Validation Suite")
    print("=" * 50 + "\n")

    tests = [
        ("Graders", test_graders),
        ("Environment Flow", test_env_flow),
        ("Determinism", test_determinism),
        ("Topics", test_topics),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
