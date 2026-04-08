import os
from crewai import Agent, LLM
from dotenv import load_dotenv
from tools import tool

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"),
)

news_researcher = Agent(
    role="senior researcher",
    goal="uncover groundbreaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of innovation, "
        "eager to explore and share knowledge that could change the world."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True,
)

news_writer = Agent(
    role="writer",
    goal="narrate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging "
        "narratives that captivate and educate, bringing new discoveries "
        "to light in an accessible manner."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=False,
)
