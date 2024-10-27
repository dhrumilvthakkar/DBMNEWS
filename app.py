import os
import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# --- API Key Configuration (Important!) ---
# Store your API key securely in Streamlit secrets.
# In your Streamlit Cloud app settings, go to "Secrets" and add
# a secret named "GOOGLE_API_KEY" with your actual API key.

# --- LLM Setup ---
async def initialize_llm():
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        verbose=True,
        temperature=0.3,
        google_api_key=google_api_key,
    )

@st.cache_resource
async def get_llm():
    return await initialize_llm()



# --- Search Tool Setup ---
search_tool = DuckDuckGoSearchRun()

# --- Streamlit App ---
st.title("Daily Tech News Generator")

topic = st.text_input("Enter your topic for the news:")
time = st.text_input("Enter the timeframe (e.g., 'past week', 'today'):")


async def generate_news(topic, time):
    # --- Agent and Task Setup ---
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Uncover cutting-edge developments in {topic} in {time}",
        backstory="""You work at a leading tech think tank. Your expertise lies in identifying emerging trends. You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        llm=await get_llm(),
        tools=[search_tool],
    )

    writer = Agent(
        role="Tech Content Strategist",
        goal=f"Craft compelling news post on {topic} advancements in {time}",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging news articles. You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=False,
        llm= await get_llm(),
        tools=[],
    )

    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the latest advancements in {topic} in {time}.
                      Identify key trends, breakthrough technologies, and potential industry impacts.
                      Your final answer MUST be a full analysis report""",
        agent=researcher,
        expected_output="Analysis research report on latest advancements",
    )

    task2 = Task(
        description=f"""Using the insights provided, develop an engaging news post that highlights the most significant {topic} advancements in {time}.
                      Your post should be informative yet accessible, catering to a tech-savvy audience.
                      Make it sound cool, avoid complex words so it doesn't sound like AI.
                      Your final answer MUST be the full news post of at least 4 paragraphs.""",
        agent=writer,
        expected_output="Full news post on significant advancements",
    )

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)
    return crew.kickoff()



async def main():
    if st.button("Generate News"):
        if topic and time:
            with st.spinner("Generating news..."):
                result = await generate_news(topic, time)

            st.subheader("Generated News:")
            st.markdown(result)
        else:
            st.warning("Please enter both a topic and a timeframe.")

if __name__ == "__main__":
    asyncio.run(main())
