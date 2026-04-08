import os
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import news_researcher, news_writer
from tasks import make_research_task, make_write_task

load_dotenv()


def run_news_crew(topic: str, constraints: dict = None) -> str:
    """
    Run the news writer crew for a given topic and optional constraints.
    Returns the generated article as a string.
    """
    research_task = make_research_task(topic)
    write_task = make_write_task(topic, constraints)

    crew = Crew(
        agents=[news_researcher, news_writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return str(result)
