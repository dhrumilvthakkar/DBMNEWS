import os
import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# --- API Key Configuration ---
# Store your API key securely in Streamlit secrets as GOOGLE_API_KEY

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

async def get_llm():  # Async but no caching
    return await initialize_llm()

# --- Initialize LLM outside Streamlit context ---
llm = None

async def setup_llm():
    global llm
    llm = await get_llm()

# --- Search Tool Setup ---
search_tool = DuckDuckGoSearchRun()

# --- Streamlit App ---
st.title("News Article Generator")

topic = st.text_input("Enter your topic for the news:")  # Only topic input

async def generate_news(topic):  # Removed time parameter
    if llm is None:
        st.error("LLM initialization failed. Check API key and logs.")
        return None

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Uncover cutting-edge developments in {topic}",  # Removed time
        backstory="""You work at a leading tech think tank. Your expertise lies in identifying emerging trends. You have a knack for dissecting complex data and presenting actionable insights, including relevant sources.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool],
    )

    writer = Agent(
        role="Tech Content Strategist",
        goal=f"Craft compelling news post on {topic} advancements",  # Removed time
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging news articles. You transform complex concepts into compelling narratives and always cite your sources.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],
    )

    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the latest advancements in {topic}, including at least 3 sources.
                      Identify key trends, breakthrough technologies, and potential industry impacts.
                      Your final answer MUST be a full analysis report with sources (URLs).""",  # Removed time
        agent=researcher,
        expected_output="Analysis research report including sources",
    )

    task2 = Task(
        description=f"""Using the insights and SOURCES provided, develop an engaging news post that highlights the most significant {topic} advancements. Include links to the sources at the end of the post.
                      Your post should be informative yet accessible, catering to a tech-savvy audience. Make it sound cool, avoid complex words.
                      The final answer MUST be the full news post with source links at the end.""",  # Removed time
        agent=writer,
        expected_output="Full news post with source links",
    )

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)
    try:
        return crew.kickoff()
    except Exception as e:
        st.error(f"An error occurred during news generation: {e}")
        return None

async def main():
    await setup_llm()

    if st.button("Generate News"):
        if topic:  # Only check for topic
            with st.spinner("Generating news..."):
                result = await generate_news(topic)  # Removed time argument
            if result:
                st.subheader("Generated News:")
                st.markdown(result)
        else:
            st.warning("Please enter a topic.") # Changed warning message


if __name__ == "__main__":
    asyncio.run(main())
