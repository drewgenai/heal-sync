import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

import chainlit as cl
import numpy as np
import pandas as pd
import shutil

from dotenv import load_dotenv
from langchain.schema.runnable.config import RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader, PyMuPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Constants
UPLOAD_PATH = "./uploads"
INITIAL_EMBEDDINGS_DIR = "./initial_embeddings"
INITIAL_EMBEDDINGS_NAME = "initial_embeddings"
XLSX_MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
PDF_MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
USER_EMBEDDINGS_NAME = "user_embeddings"

# Make sure upload directory exists
os.makedirs(UPLOAD_PATH, exist_ok=True)

# NIH HEAL CDE core domains
NIH_HEAL_DOMAINS = [
    "Pain intensity",
    "Pain interference",
    "Physical functioning/quality of life (QoL)",
    "Sleep",
    "Pain catastrophizing",
    "Depression",
    "Anxiety",
    "Global satisfaction with treatment",
    "Substance Use Screener",
    "Quality of Life (QoL)"
]

# Initialize Qdrant (in-memory)
qdrant_client = QdrantClient(":memory:")

# Create a semantic splitter for PDF documents
semantic_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# Utility functions
def load_and_chunk_excel_files():
    """Loads all .xlsx files from the initial embeddings directory and splits them into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_chunks = []
    file_count = 0
    
    print("Processing Excel files...")
    
    for file in os.listdir(INITIAL_EMBEDDINGS_DIR):
        if file.endswith(".xlsx"):
            file_path = os.path.join(INITIAL_EMBEDDINGS_DIR, file)
            
            try:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
                
                chunks = text_splitter.split_documents(documents)
                
                for chunk in chunks:
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata["filename"] = file
                
                all_chunks.extend(chunks)
                file_count += 1
                print(f"Processed: {file} - {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    print(f"Processed {file_count} Excel files with a total of {len(all_chunks)} chunks.")
    return all_chunks

def embed_chunks_in_qdrant(chunks):
    """Embeds document chunks and stores them in Qdrant."""
    if not chunks:
        print("No Excel files found to process or all files were empty.")
        return None

    xlsx_model = HuggingFaceEmbeddings(model_name=XLSX_MODEL_ID)
    print("Creating vector store...")
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=xlsx_model,
        location=":memory:",
        collection_name=INITIAL_EMBEDDINGS_NAME
    )
    print(f"Successfully loaded all .xlsx files into Qdrant collection '{INITIAL_EMBEDDINGS_NAME}'.")
    return vector_store

def process_initial_embeddings():
    """Loads all .xlsx files, extracts text, embeds, and stores in Qdrant."""
    chunks = load_and_chunk_excel_files()
    return embed_chunks_in_qdrant(chunks)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def load_and_chunk_pdf_files(files):
    """Load PDF files and split them into chunks with metadata."""
    print(f"Loading {len(files)} uploaded PDF files")
    documents_with_metadata = []
    
    for file in files:
        print(f"Processing file: {file.name}, size: {file.size} bytes")
        file_path = os.path.join(UPLOAD_PATH, file.name)
        
        # Ensure the upload directory exists
        os.makedirs(UPLOAD_PATH, exist_ok=True)
        
        # Copy the file to the upload directory
        shutil.copyfile(file.path, file_path)
        
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                source_name = file.name
                chunks = semantic_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    doc_chunk = Document(page_content=chunk, metadata={"source": source_name})
                    documents_with_metadata.append(doc_chunk)
                    
            print(f"Successfully processed {file.name}, extracted {len(documents_with_metadata)} chunks")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    return documents_with_metadata

async def embed_pdf_chunks_in_qdrant(documents_with_metadata, model_name=PDF_MODEL_ID):
    """Create a vector store and embed PDF chunks into Qdrant."""
    if not documents_with_metadata:
        print("No documents to embed")
        return None
        
    # Create a new embeddings model
    pdf_model = HuggingFaceEmbeddings(model_name=model_name)
    
    try:
        # First, check if collection exists and delete it if it does
        if USER_EMBEDDINGS_NAME in [c.name for c in qdrant_client.get_collections().collections]:
            qdrant_client.delete_collection(USER_EMBEDDINGS_NAME)
            
        # Create the collection with proper parameters
        # Get the embedding dimension by creating a sample embedding
        sample_text = "Sample text to determine embedding dimension"
        sample_embedding = pdf_model.embed_query(sample_text)
        embedding_dimension = len(sample_embedding)
        
        qdrant_client.create_collection(
            collection_name=USER_EMBEDDINGS_NAME,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        
        # Create the vector store
        user_vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=pdf_model
        )
        
        # Add documents to the vector store
        user_vectorstore.add_documents(documents_with_metadata)
        
        print(f"Added {len(documents_with_metadata)} chunks from uploaded files to collection '{USER_EMBEDDINGS_NAME}'")
        return user_vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

async def process_uploaded_files(files, model_name=PDF_MODEL_ID):
    """Process uploaded PDF files and add them to a separate vector store collection"""
    documents_with_metadata = await load_and_chunk_pdf_files(files)
    return await embed_pdf_chunks_in_qdrant(documents_with_metadata, model_name)

# Data processing and initialization
vectorstore = process_initial_embeddings()


# Create a retriever from the vector store
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("Retriever created successfully.")
else:
    print("Failed to create retriever: No vector store available.")

naive_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

# RAG setup
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chat_model = ChatOpenAI()

initialembeddings_retrieval_chain = (
    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question")}
    | rag_prompt 
    | chat_model
    | StrOutputParser()
)

# Tool definitions
@tool
def search_excel_data(query: str, top_k: int = 3) -> str:
    """Search both Excel data and user-uploaded PDF data for information related to the query.
    
    Args:
        query: The search query
        top_k: Number of results to return (default: 3)
        
    Returns:
        String containing the search results with their content and source files
    """
    # Use the existing initialembeddings_retrieval_chain
    result = initialembeddings_retrieval_chain.invoke({"question": query})
    
    # If we have a user collection, also search that
    try:
        user_vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=HuggingFaceEmbeddings(model_name=XLSX_MODEL_ID)
        )
        
        # Create a retrieval chain for user documents
        user_retriever = user_vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        user_retrieval_chain = (
            {"context": itemgetter("question") | user_retriever | format_docs, 
             "question": itemgetter("question")}
            | rag_prompt 
            | chat_model
            | StrOutputParser()
        )
        
        user_result = user_retrieval_chain.invoke({"question": query})
        
        # Combine results
        return f"From Excel files:\n{result}\n\nFrom your uploaded PDF:\n{user_result}"
    except Exception as e:
        # If no user collection exists yet, just return Excel results
        return result

@tool
def identify_heal_instruments(protocol_text: str = "") -> str:
    """Identify instruments used in the protocol for each NIH HEAL CDE core domain.
    
    Args:
        protocol_text: Optional text from the protocol to analyze
        
    Returns:
        String containing identified instruments for each domain
    """
    # Check if user collection exists
    try:
        # Check if files exist in the upload directory
        uploaded_files = [f for f in os.listdir(UPLOAD_PATH) if f.endswith('.pdf')]
        
        if not uploaded_files:
            return "No protocol document has been uploaded yet."
            
        user_vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=HuggingFaceEmbeddings(model_name=XLSX_MODEL_ID)
        )
        user_retriever = user_vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        print(f"Error accessing user vector store: {str(e)}")
        return "No protocol document has been uploaded yet or there was an error accessing it."
    
    # For each domain, search for relevant instruments
    domain_instruments = {}
    
    for domain in NIH_HEAL_DOMAINS:
        # Search for instruments related to this domain in the protocol
        query = f"What instrument or measure is used for {domain} in the protocol?"
        
        # Retrieve relevant chunks from the protocol
        try:
            docs = user_retriever.invoke(query)
            protocol_context = format_docs(docs)
            
            # Search for instruments in the Excel data that match this domain
            excel_query = f"What are standard instruments or measures for {domain}?"
            excel_instruments = initialembeddings_retrieval_chain.invoke({"question": excel_query})
            
            # Use the model to identify the most likely instrument for this domain
            prompt = f"""
            Based on the protocol information and known instruments, identify which instrument is being used for the domain: {domain}
            
            Protocol information:
            {protocol_context}
            
            Known instruments for this domain:
            {excel_instruments}
            
            Respond with only the name of the identified instrument. If you cannot identify a specific instrument, respond with "Not identified".
            """
            
            instrument = chat_model.invoke([HumanMessage(content=prompt)]).content
            domain_instruments[domain] = instrument.strip()
            print(f"Identified instrument for {domain}: {instrument.strip()}")
        except Exception as e:
            print(f"Error identifying instrument for {domain}: {str(e)}")
            domain_instruments[domain] = "Error during identification"
    
    # Format the results as a markdown table
    result = "# NIH HEAL CDE Core Domains and Identified Instruments\n\n"
    result += "| Domain | Protocol Instrument |\n"
    result += "|--------|--------------------|\n"
    
    for domain, instrument in domain_instruments.items():
        result += f"| {domain} | {instrument} |\n"
    
    return result

tools = [search_excel_data, identify_heal_instruments]

# LangGraph components
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
final_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# System message for the model
system_message = """You are a helpful assistant specializing in NIH HEAL CDE protocols.

You have access to:
1. Excel data through the search_excel_data tool
2. A tool to identify instruments in NIH HEAL protocols (identify_heal_instruments)

WHEN TO USE TOOLS:
- When users ask about instruments, measures, assessments, questionnaires, or scales in a protocol, use the identify_heal_instruments tool.
- When users ask about data or information in the Excel files, use the search_excel_data tool.
- For general questions about NIH HEAL CDE domains, use the search_excel_data tool.

Be specific in your tool queries to get the most relevant information.
Always use the appropriate tool before responding to questions about the protocol or Excel data.
"""

# Bind tools and configure models
model = model.bind_tools(tools)
final_model = final_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "final"

def call_model(state: MessagesState):
    messages = state["messages"]
    # Add the system message at the beginning of the messages list
    if messages and not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def call_final_model(state: MessagesState):
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = final_model.invoke(
        [
            SystemMessage("Rewrite this in the voice of a helpful and kind assistant"),
            HumanMessage(last_ai_message.content),
        ]
    )
    # overwrite the last AI message from the agent
    response.id = last_ai_message.id
    return {"messages": [response]}

# Build the graph
builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")
builder.add_edge("final", END)

graph = builder.compile()

# Chainlit handlers
@cl.on_chat_start
async def on_chat_start():
    # Welcome message
    welcome_msg = cl.Message(content="Welcome! Please upload a NIH HEAL protocol PDF file to get started.")
    await welcome_msg.send()
    
    # Wait for file upload
    files = await cl.AskFileMessage(
        content="Please upload a NIH HEAL protocol PDF file to analyze alongside the Excel data.",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()
    
    if files:
        processing_msg = cl.Message(content="Processing your protocol PDF file...")
        await processing_msg.send()
        
        # Process the uploaded files
        user_vectorstore = await process_uploaded_files(files)
        
        if user_vectorstore:
            analysis_msg = cl.Message(content="Analyzing your protocol to identify instruments for NIH HEAL CDE core domains...")
            await analysis_msg.send()
            
            # Use the identify_heal_instruments tool to analyze the protocol
            config = {"configurable": {"thread_id": cl.context.session.id}}
            
            # Create a message to trigger the analysis
            analysis_request = HumanMessage(content="Please analyze the uploaded protocol and identify instruments for each NIH HEAL CDE core domain.")
            
            final_answer = cl.Message(content="")
            
            for msg, metadata in graph.stream(
                {"messages": [analysis_request]}, 
                stream_mode="messages", 
                config=config
            ):
                if (
                    msg.content
                    and not isinstance(msg, HumanMessage)
                    and metadata["langgraph_node"] == "final"
                ):
                    await final_answer.stream_token(msg.content)
            
            await final_answer.send()
            
            await cl.Message(content="You can now ask additional questions about the protocol or the Excel data.").send()
        else:
            await cl.Message(content="There was an issue processing your PDF. Please try uploading again.").send()
    else:
        await cl.Message(content="No file was uploaded. You can still ask questions about the Excel data.").send()

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    
    # Completely disable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    
    # For all messages, use the graph to handle the logic
    final_answer = cl.Message(content="")
    
    # Check if files exist for instrument-related queries
    if (any(keyword in msg.content.lower() for keyword in ["instrument", "measure", "assessment", "questionnaire", "scale", "protocol"]) and 
        not any(f for f in os.listdir(UPLOAD_PATH) if f.endswith('.pdf'))):
        await cl.Message(content="No protocol document has been detected. Please upload a protocol document first.").send()
        return
    
    # Let the graph handle all message processing
    for msg_response, metadata in graph.stream(
        {"messages": [HumanMessage(content=msg.content)]}, 
        stream_mode="messages", 
        config=config
    ):
        if (
            msg_response.content
            and not isinstance(msg_response, HumanMessage)
            and metadata["langgraph_node"] == "final"
        ):
            await final_answer.stream_token(msg_response.content)

    await final_answer.send()