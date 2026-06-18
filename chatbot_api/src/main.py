from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from agents.hospital_rag_agent import hospital_rag_agent
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from utils.async_utils import async_retry

app=FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot"
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await hospital_rag_agent.ainvoke({"messages":[HumanMessage(content=query)]})

@app.get("/")
async def get_status():
    return {"status":"running"}

@app.post("/hospital-rag-agent")
async def query_hospital_agent(query: HospitalQueryInput) -> HospitalQueryOutput:
    query_response=await invoke_agent_with_retry(query.text)
    final_output=query_response["messages"][-1].content

    steps=[]
    for msg in query_response["messages"]:
        tool_calls=getattr(msg, "tool_calls", None)
        if tool_calls:
            for tool in tool_calls:
                steps.append(f"Invoked Tool: {tool['name']} with args {tool['args']}")
    
    return {
        "input":query.text,
        "output":final_output,
        "intermediate_steps":steps
    }