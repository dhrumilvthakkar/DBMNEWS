import os
import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# --- Text Splitter for RAG ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- Streamlit App ---
st.title("Article Generator on all News Facts about Any Topic")

topic = st.text_input("Enter your topic for the news:")

async def generate_news(topic):
    if llm is None:
        st.error("LLM initialization failed. Check API key and logs.")
        return None

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Uncover cutting-edge developments in {topic}",
        backstory=""" You work at a leading tech think tank. Your expertise lies in identifying emerging trends. You have a knack for dissecting complex data and presenting actionable insights, including relevant sources.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool],
    )

    writer = Agent(
        role="Tech Content Strategist",
        goal=f"Craft compelling news post on {topic} advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging news articles. You transform complex concepts into compelling narratives and always cite your sources.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],
    )

    task1 = Task(
        description=f"""Conduct a comprehensive analysis of the latest advancements in {topic}, including at least 3 sources.
                      For each source, extract relevant excerpts (maximum 200 words each) related to the topic.
                      Identify key trends, breakthrough technologies, and potential industry impacts.
                      Your final answer MUST be a full analysis report with excerpts from sources and URLs.""",
        agent=researcher,
        expected_output="Analysis research report including excerpts and sources",
    )

    task2 = Task(
        description=f"""Using the insights and EXCERPTS provided, develop an engaging news post that highlights the most significant {topic} advancements. Include links to the sources at the end of the post.  Quote the excerpts directly within the news post.
                      Your post should be informative, accessible, and cater to a tech-savvy audience. Make it sound cool, avoid complex words.
                      The final answer MUST be the full news post with quoted excerpts and source links at the end.""",
        agent=writer,
        expected_output="Full news post with excerpts and source links",
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=1,
        summarization_chain=chain,
        text_splitter=text_splitter,
    )
    try:
        return crew.kickoff()
    except Exception as e:
        st.error(f"An error occurred during news generation: {e}")
        return None


async def main():
    await setup_llm()

    if st.button("Generate News"):
        if topic:
            with st.spinner("Generating news..."):
                result = await generate_news(topic)
            if result:
                st.subheader("Generated News:")
                st.markdown(result)
        else:
            st.warning("Please enter a topic.")

if __name__ == "__main__":
    asyncio.run(main())
