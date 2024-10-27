import os
import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# --- API Key Configuration ---
# Store your API key securely in Streamlit secrets.

# --- LLM Setup ---
async def initialize_llm():
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            verbose=True,
            temperature=0.3,
            google_api_key=google_api_key,
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

@st.cache_resource
async def get_llm():
    return await initialize_llm()

# --- Global LLM instance ---
llm_instance = None

async def get_llm_outside_generate_news():
    global llm_instance
    if llm_instance is None:
        llm_instance = await get_llm()
    return llm_instance

# --- Search Tool Setup ---
search_tool = DuckDuckGoSearchRun()

# --- Streamlit App ---
st.title("Daily Tech News Generator")

topic = st.text_input("Enter your topic for the news:")
time = st.text_input("Enter the timeframe (e.g., 'past week', 'today'):")

async def generate_news(topic, time):
    llm = await get_llm_outside_generate_news()
    if llm is None:
        st.error("LLM initialization failed. Please check your API key and logs.")
        return

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Uncover cutting-edge developments in {topic} in {time}",
        backstory="""...""",  # Add your backstory here
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool],
    )

    writer = Agent(
        role="Tech Content Strategist",
        goal=f"Craft compelling news post on {topic} advancements in {time}",
        backstory="""...""", # Add your backstory here
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],
    )

    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the latest advancements in {topic} in {time}, including sources.
                      Identify key trends, breakthrough technologies, and potential industry impacts.
                      Your final answer MUST be a full analysis report with sources.""",
        agent=researcher,
        expected_output="Analysis research report including sources",
    )

    task2 = Task(
        description=f"""Using the insights and SOURCES provided, develop an engaging news post that highlights the most significant {topic} advancements in {time}.  Include links to the sources at the end of the post.
                      Your post should be informative yet accessible, catering to a tech-savvy audience. Make it sound cool, avoid complex words.
                      The final answer MUST be the full news post with source links at the end.""",
        agent=writer,
        expected_output="Full news post with source links",
    )

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)
    return crew.kickoff()

async def main():
    await get_llm_outside_generate_news() # Initialize LLM at app start

    if st.button("Generate News"):
        if topic and time:
            with st.spinner("Generating news..."):
                result = await generate_news(topic, time)
            if result:  # Check if result is not None (in case of errors)
                st.subheader("Generated News:")
                st.markdown(result)
        else:
            st.warning("Please enter both a topic and a timeframe.")


if __name__ == "__main__":
    asyncio.run(main())
