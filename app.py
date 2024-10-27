# -*- coding: utf-8 -*-

import os
import streamlit as st
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# --- API Key Configuration (Important!) ---
# Store your API key securely, e.g., in environment variables or secrets management.
GOOGLE_API_KEY = st.secrets("GOOGLE_API_KEY")  # Recommended approach

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
)

# --- Search Tool Setup ---
search_tool = DuckDuckGoSearchRun()


# --- Streamlit App ---
st.title("Daily Tech News Generator")

topic = st.text_input("Enter your topic for the news:")
time = st.text_input("Enter the timeframe (e.g., 'past week', 'today'):")

if st.button("Generate News"):
    if topic and time:
        with st.spinner("Generating news..."):
            # --- Agent and Task Setup ---
            researcher = Agent(
                role="Senior Research Analyst",
                goal=f"Uncover cutting-edge developments in {topic} in {time}",
                backstory="""You work at a leading tech think tank. Your expertise lies in identifying emerging trends. You have a knack for dissecting complex data and presenting actionable insights.""",
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[search_tool],
            )

            writer = Agent(
                role="Tech Content Strategist",
                goal=f"Craft compelling news post on {topic} advancements in {time}",
                backstory="""You are a renowned Content Strategist, known for your insightful and engaging news articles. You transform complex concepts into compelling narratives.""",
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[],
            )

            task1 = Task(
                description=f"""Conduct a comprehensive analysis of the latest news on {topic} in {time}.
                              Identify key trends, breakthrough technologies, and potential industry impacts.
                              Your final answer MUST be a full analysis report""",
                agent=researcher,
                expected_output="Analysis research report on latest advancements", # More general
            )

            task2 = Task(
                description=f"""Using the insights provided, develop an engaging news post that highlights the most significant {topic} advancements in {time}.
                              Your post should be informative yet accessible, catering to a tech-savvy audience.
                              Make it sound cool, avoid complex words so it doesn't sound like AI.
                              Your final answer MUST be the full news post of at least 4 paragraphs.""",
                agent=writer,
                expected_output="Full news post on significant advancements", # More General
            )

            crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)
            result = crew.kickoff()

        st.subheader("Generated News:")
        st.markdown(result)
    else:
        st.warning("Please enter both a topic and a timeframe.")
