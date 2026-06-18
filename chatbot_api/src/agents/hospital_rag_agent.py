import os
from chains.hospital_cypher_chain import hospital_cypher_chain
from chains.hospital_review_chain import rag_chain
from langchain.agents import create_agent
from langchain_core.tools import tool
from tools.wait_times import get_current_wait_times, get_most_available_hospital
from langchain_google_genai import ChatGoogleGenerativeAI

HOSPITAL_AGENT_MODEL=os.getenv("HOSPITAL_AGENT_MODEL")

@tool("Experiences")
def experiences(query: str) -> str:
    """Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """
    return rag_chain.invoke(query)

@tool("Graph")
def graph(query: str) -> str:
    """Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """
    return hospital_cypher_chain.invoke(query)
@tool("Waits")
def waits(hospital_name: str) -> str:
    """Use when asked about current wait times
    at a specific hospital. This tool can only get the current
    wait time at a hospital and does not have any information about
    aggregate or historical wait times. Do not pass the word "hospital"
    as input, only the hospital name itself. For example, if the prompt
    is "What is the current wait time at Jordan Inc Hospital?", the
    input should be "Jordan Inc".
    """
    return get_current_wait_times(hospital_name)
 
 
@tool("Availability")
def availability() -> dict:
    """
    Use when you need to find out which hospital has the shortest
    wait time. This tool does not have any information about aggregate
    or historical wait times. This tool returns a dictionary with the
    hospital name as the key and the wait time in minutes as the value.
    """
    return get_most_available_hospital(None)

tools=[experiences, graph, waits, availability]

chat_model=ChatGoogleGenerativeAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0
)

hospital_rag_agent=create_agent(
    model=chat_model,
    tools=tools,
    system_prompt="You are a helpful hospital routing assistant. Use the provided tools to answer the user's questions."
)
