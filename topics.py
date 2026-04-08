import random

# Easy topics - well-known, lots of available content
EASY_TOPICS = [
    "AI in healthcare",
    "electric vehicles",
    "social media and mental health",
    "remote work technology",
    "smartphone innovations",
    "streaming services and entertainment",
    "cloud computing",
    "cybersecurity basics",
    "e-commerce growth",
    "renewable energy trends",
    "artificial intelligence in education",
    "5G technology",
    "smart home devices",
    "food delivery technology",
    "fitness wearables",
]

# Medium topics - require more specific knowledge + constraints applied
MEDIUM_TOPICS = [
    "quantum computing in finance",
    "CRISPR gene editing applications",
    "autonomous vehicle challenges",
    "blockchain in supply chain",
    "edge computing and IoT",
    "digital mental health platforms",
    "AI-generated content moderation",
    "green hydrogen energy",
    "augmented reality in retail",
    "precision agriculture technology",
    "neurotechnology and brain-computer interfaces",
    "decentralized finance (DeFi)",
    "synthetic biology breakthroughs",
    "satellite internet expansion",
    "AI in drug discovery",
]

# Hard topics - niche, controversial, or require structured investigative writing
HARD_TOPICS = [
    "algorithmic bias in hiring systems",
    "AI regulation and governance frameworks",
    "deepfake detection technology",
    "data sovereignty and digital nationalism",
    "AI consciousness and sentience debate",
    "geoengineering to combat climate change",
    "longevity biotech and anti-aging research",
    "AI-driven surveillance and civil liberties",
    "the future of nuclear fusion energy",
    "digital identity and self-sovereign identity",
    "post-quantum cryptography",
    "the ethics of autonomous weapons",
    "AI and the future of creative professions",
    "universal basic income in an automated economy",
    "microplastics and nanotechnology solutions",
]

# Constraints for medium tasks
MEDIUM_CONSTRAINTS = [
    {"keywords": ["innovation", "impact", "future"], "min_words": 300, "max_words": 320, "required_quotes": 2, "exact_paragraphs": 5},
    {"keywords": ["challenge", "solution", "industry"], "min_words": 300, "max_words": 320, "required_quotes": 2, "exact_paragraphs": 5},
    {"keywords": ["technology", "growth", "adoption"], "min_words": 300, "max_words": 320, "required_quotes": 2, "exact_paragraphs": 5},
    {"keywords": ["research", "development", "market"], "min_words": 300, "max_words": 320, "required_quotes": 2, "exact_paragraphs": 5},
    {"keywords": ["benefits", "risks", "opportunity"], "min_words": 300, "max_words": 320, "required_quotes": 2, "exact_paragraphs": 5},
]

# Constraints for hard tasks
HARD_CONSTRAINTS = {
    "sections": ["Introduction", "Current Trends", "Challenges", "Future Outlook"],
    "require_stats": True,
    "min_words": 500,
    "max_words": 510,
    "exact_citations": 3,
    "title_length": 6,
    "exact_word_frequency": {"technology": 5},
    "exact_nth_word": {"word": "breakthrough", "n": 250},
}


def sample_easy() -> dict:
    topic = random.choice(EASY_TOPICS)
    return {"topic": topic, "constraints": {"starting_phrase": "In recent news,"}}


def sample_medium() -> dict:
    topic = random.choice(MEDIUM_TOPICS)
    constraints = random.choice(MEDIUM_CONSTRAINTS)
    return {"topic": topic, "constraints": constraints}


def sample_hard() -> dict:
    topic = random.choice(HARD_TOPICS)
    return {"topic": topic, "constraints": HARD_CONSTRAINTS}
