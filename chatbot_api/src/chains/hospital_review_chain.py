import os
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

HOSPITAL_QA_MODEL=os.getenv("HOSPITAL_QA_MODEL")
gemini_embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
local_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
neo4j_vector_index=Neo4jVector.from_existing_graph(
    embedding=local_embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_USERNAME"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name"
    ],
    embedding_node_property="embedding",
)
llm=ChatGoogleGenerativeAI(model=os.getenv("HOSPITAL_QA_MODEL"), temperature=0)
retriever=neo4j_vector_index.as_retriever(k=12)

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_prompt=ChatPromptTemplate.from_messages([
    ("system", review_template),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context": retriever, "question":RunnablePassthrough()}
    | review_prompt
    | llm
    | StrOutputParser()
)