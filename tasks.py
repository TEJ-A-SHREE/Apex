from crewai import Task
from tools import tool
from agents import news_researcher, news_writer


def make_research_task(topic: str) -> Task:
    return Task(
        description=(
            f"Identify the next big trend in {topic}. "
            "Focus on identifying pros and cons and the overall narrative. "
            "Your final report should clearly articulate the key points, "
            "its market opportunities, and potential risks."
        ),
        expected_output="A comprehensive 3 paragraphs long report on the latest trends.",
        tools=[tool],
        agent=news_researcher,
    )


def make_write_task(topic: str, constraints: dict = None) -> Task:
    constraints = constraints or {}
    constraint_text = ""

    if constraints.get("keywords"):
        kw = ", ".join(constraints["keywords"])
        constraint_text += f" You MUST include the following keywords: {kw}."

    if constraints.get("min_words") and constraints.get("max_words"):
        constraint_text += (
            f" The article MUST be between {constraints['min_words']} "
            f"and {constraints['max_words']} words."
        )

    if constraints.get("sections"):
        sections = ", ".join(constraints["sections"])
        constraint_text += (
            f" The article MUST contain these sections as headings: {sections}."
        )

    if constraints.get("require_stats"):
        constraint_text += (
            " The article MUST include at least 2 specific facts or statistics "
            "(numbers, percentages, or data points)."
        )

    return Task(
        description=(
            f"Compose an insightful article on {topic}. "
            "Focus on the latest trends and how it's impacting the industry. "
            "This article should be easy to understand, engaging, and positive."
            + constraint_text
        ),
        expected_output=(
            f"A well-structured article on {topic} formatted as markdown."
        ),
        tools=[tool],
        agent=news_writer,
        async_execution=False,
    )
