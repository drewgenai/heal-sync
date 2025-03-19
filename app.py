import os
import csv

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
from langchain_experimental.text_splitter import SemanticChunker

# Load environment variables
load_dotenv()
#####LangSmith uncomment block for LangSmith tracing
# import uuid
# os.environ["LANGCHAIN_PROJECT"] = f"HEAL-SYNC - {uuid.uuid4().hex[0:8]}"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# print(os.environ["LANGCHAIN_PROJECT"])
###########LangSmith


# ==================== CONSTANTS ====================
# Paths and directories
UPLOAD_PATH = "./uploads"
INITIAL_EMBEDDINGS_DIR = "./initial_embeddings"
INITIAL_EMBEDDINGS_NAME = "initial_embeddings"
USER_EMBEDDINGS_NAME = "user_embeddings"

os.makedirs(UPLOAD_PATH, exist_ok=True)

# Model IDs
EMBEDDING_MODEL_ID = "pritamdeka/S-PubMedBert-MS-MARCO"
#EMBEDDING_MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
general_llm_model = "gpt-4o"
chat_model = ChatOpenAI(model_name=general_llm_model)

# NIH HEAL CDE core domains
NIH_HEAL_CORE_DOMAINS = [
    "Anxiety",
    "Depression",
    "Global satisfaction with treatment",
    "Pain catastrophizing",
    "Pain interference",
    "Pain intensity",
    "Physical functioning",
    "Quality of Life (QoL)",
    "Sleep",
    "Substance Use Screener"
]


# ==================== EMBEDDING MODEL SETUP to allow flexibility of model selection ====================
def get_embedding_model(model_id):
    """Creates and returns the appropriate embedding model based on the model ID."""
    if "text-embedding" in model_id:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_id)
    else:
        return HuggingFaceEmbeddings(model_name=model_id)

def initialize_embedding_models():
    """Initialize a single embedding model for all document types"""
    global embedding_model
    
    # Initialize a single model for all document types
    embedding_model = get_embedding_model(EMBEDDING_MODEL_ID)
    
    print(f"Initialized embedding model: {EMBEDDING_MODEL_ID}")

# Initialize the embedding model
initialize_embedding_models()

# ==================== QDRANT SETUP ====================
# Create a global Qdrant client for the core embeddings (available to all sessions)
global_qdrant_client = QdrantClient(":memory:")

# Initialize a function to create session-specific Qdrant clients
def create_session_qdrant_client():
    return QdrantClient(":memory:")

# ==================== DOCUMENT PROCESSING ====================
# Create a semantic splitter for documents
semantic_splitter = SemanticChunker(embedding_model, add_start_index=True, buffer_size=30)
# Keep the recursive splitter as a fallback option
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def format_docs(docs):
    """Format a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ==================== CORE EMBEDDINGS PROCESSING ====================
def load_and_chunk_core_reference_files():
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
                    chunk.metadata["type"] = "excel"
                
                all_chunks.extend(chunks)
                file_count += 1
                print(f"Processed: {file} - {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    print(f"Processed {file_count} Excel files with a total of {len(all_chunks)} chunks.")
    return all_chunks

def embed_core_reference_in_qdrant(chunks):
    """Embeds core reference chunks and stores them in the global Qdrant instance."""
    global embedding_model, global_qdrant_client
    
    if not chunks:
        print("No Excel files found to process or all files were empty.")
        return None

    # Ensure embedding model is valid
    if embedding_model is None:
        print("ERROR: No embedding model available. Initializing now.")
        initialize_embedding_models()
        
    print(f"Using embedding model: {EMBEDDING_MODEL_ID}")
    print("Creating vector store for core reference data...")
    
    try:
        # First, check if collection exists and delete it if it does
        if INITIAL_EMBEDDINGS_NAME in [c.name for c in global_qdrant_client.get_collections().collections]:
            global_qdrant_client.delete_collection(INITIAL_EMBEDDINGS_NAME)
            
        # Create the collection with proper parameters
        # Get the embedding dimension from the model
        embedding_dimension = len(embedding_model.embed_query("Sample text"))
        
        global_qdrant_client.create_collection(
            collection_name=INITIAL_EMBEDDINGS_NAME,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        
        # Create the vector store
        vector_store = QdrantVectorStore(
            client=global_qdrant_client,
            collection_name=INITIAL_EMBEDDINGS_NAME,
            embedding=embedding_model
        )
        
        # Add documents to the vector store
        vector_store.add_documents(chunks)
        
        print(f"Successfully loaded all .xlsx files into Qdrant collection '{INITIAL_EMBEDDINGS_NAME}'.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        print(f"Embedding model status: {embedding_model is not None}")
        return None

# ==================== RAG SETUP ====================
# RAG template for all retrievals
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)


# Initialize core embeddings on application startup
core_vectorstore = None
core_retriever = None  # Global core retriever

def initialize_core_reference_embeddings():
    """Loads all .xlsx files, extracts text, embeds, and stores in global Qdrant."""
    global core_vectorstore, core_retriever
    chunks = load_and_chunk_core_reference_files()
    core_vectorstore = embed_core_reference_in_qdrant(chunks)
    
    # Create the core retriever if vector store was created successfully
    if core_vectorstore:
        core_retriever = core_vectorstore.as_retriever(search_kwargs={"k": 10})
        print("Core reference retriever created successfully.")
    else:
        print("Failed to create core reference retriever: No vector store available.")
        
    return core_vectorstore

# Initialize core reference embeddings
core_vectorstore = initialize_core_reference_embeddings()

# Chain for retrieving from core reference embeddings
if core_retriever:
    core_reference_retrieval_chain = (
        {"context": itemgetter("question") | core_retriever | format_docs, 
         "question": itemgetter("question")}
        | rag_prompt 
        | chat_model
        | StrOutputParser()
    )

# ==================== PROTOCOL DOCUMENT PROCESSING ====================
async def load_and_chunk_protocol_files(files):
    """Load protocol PDF files and split them into chunks with metadata."""
    print(f"Loading {len(files)} uploaded PDF files")
    documents_with_metadata = []
    
    for file in files:
        print(f"Processing file: {file.name}, size: {file.size} bytes")
        file_path = os.path.join(UPLOAD_PATH, file.name)
        
        shutil.copyfile(file.path, file_path)
        
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Add source filename to metadata for all documents
            for doc in documents:
                doc.metadata["source"] = file.name
                doc.metadata["type"] = "pdf"
            
            # Use semantic_splitter.split_documents to preserve metadata
            chunks = semantic_splitter.split_documents(documents)
            documents_with_metadata.extend(chunks)
                    
            print(f"Successfully processed {file.name}, extracted {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            # Fallback to recursive splitter if semantic chunking fails
            try:
                print(f"Falling back to recursive character splitting for {file.name}")
                chunks = recursive_splitter.split_documents(documents)
                documents_with_metadata.extend(chunks)
                print(f"Fallback successful, extracted {len(chunks)} chunks")
            except Exception as fallback_error:
                print(f"Fallback also failed: {str(fallback_error)}")
    
    return documents_with_metadata

async def embed_protocol_in_qdrant(documents_with_metadata, session_qdrant_client, model_name=EMBEDDING_MODEL_ID):
    """Create a vector store and embed protocol chunks into session-specific Qdrant."""
    global embedding_model
      
    print(f"Using embedding model: {model_name}")
    
    try:
        # First, check if collection exists and delete it if it does
        if USER_EMBEDDINGS_NAME in [c.name for c in session_qdrant_client.get_collections().collections]:
            session_qdrant_client.delete_collection(USER_EMBEDDINGS_NAME)
            
        # Create the collection with proper parameters
        # Get the embedding dimension from the model
        embedding_dimension = len(embedding_model.embed_query("Sample text"))
        
        session_qdrant_client.create_collection(
            collection_name=USER_EMBEDDINGS_NAME,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        
        # Create the vector store
        user_vectorstore = QdrantVectorStore(
            client=session_qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=embedding_model
        )
        
        # Add documents to the vector store
        user_vectorstore.add_documents(documents_with_metadata)
        
        print(f"Added {len(documents_with_metadata)} chunks from uploaded files to collection '{USER_EMBEDDINGS_NAME}'")
        return user_vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

async def process_uploaded_protocol(files, session_qdrant_client, model_name=EMBEDDING_MODEL_ID):
    """Process uploaded protocol PDF files and add them to a session-specific vector store collection"""
    documents_with_metadata = await load_and_chunk_protocol_files(files)
    return await embed_protocol_in_qdrant(documents_with_metadata, session_qdrant_client, model_name)

# ==================== RETRIEVAL FUNCTIONS ====================
async def retrieve_from_core(query, k=5):
    """Retrieve documents from core reference database"""
    global core_retriever
    
    if not core_retriever:
        print("No core retriever available")
        return []
    
    # Override k if needed
    if k != 10:  # Assuming default k=10 was used when creating the retriever
        retriever = core_vectorstore.as_retriever(search_kwargs={"k": k})
        return await retriever.ainvoke(query)
    
    return await core_retriever.ainvoke(query)

async def retrieve_from_protocol(query, k=5):
    """Retrieve documents from protocol database"""
    # Get the session-specific client
    session_qdrant_client = cl.user_session.get("session_qdrant_client")
    if not session_qdrant_client:
        print("No session client available")
        return []
    
    # Check if collection exists
    try:
        if USER_EMBEDDINGS_NAME not in [c.name for c in session_qdrant_client.get_collections().collections]:
            print("No protocol document embedded")
            return []
    except Exception as e:
        print(f"Error checking collections: {str(e)}")
        return []
    
    # Create vector store with the session client
    protocol_vectorstore = QdrantVectorStore(
        client=session_qdrant_client,
        collection_name=USER_EMBEDDINGS_NAME,
        embedding=embedding_model
    )
    
    # Create and use retriever
    protocol_retriever = protocol_vectorstore.as_retriever(search_kwargs={"k": k})
    return await protocol_retriever.ainvoke(query)

# ==================== TOOL DEFINITIONS ====================
@tool
async def search_all_data(query: str, doc_type: str = None) -> str:
    """Search all data or filter by document type (protocol/core_reference)"""
    try:
        chain = await create_rag_chain(doc_type)
        return await chain.ainvoke({"question": query})
    except Exception as e:
        return f"Error searching data: {str(e)}"

@tool
async def analyze_protocol_domains(export_csv: bool = True) -> str:
    """Analyze all NIH HEAL CDE core domains and identify instruments used in the protocol(s)."""
    # Check if protocol document exists
    session_qdrant_client = cl.user_session.get("session_qdrant_client")
    if not session_qdrant_client or USER_EMBEDDINGS_NAME not in [c.name for c in session_qdrant_client.get_collections().collections]:
        return "No protocol document has been uploaded yet."
    
    # Get the names of the uploaded protocol files from the user session
    protocol_names = cl.user_session.get("protocol_filenames", ["Unknown Protocol"])
    
    # Use asyncio.gather to run all domain searches in parallel
    import asyncio
    tasks = [_search_protocol_for_instruments(domain) for domain in NIH_HEAL_CORE_DOMAINS]
    results = await asyncio.gather(*tasks)
    
    # Format the results as a markdown table
    title = "NIH HEAL CDE Core Domains Analysis"
    result = f"# {title}\n\n"
    
    # Create header with all protocol names
    result += "| Domain |"
    for protocol_name in protocol_names:
        result += f" Protocol Instrument - {protocol_name} |"
    result += "\n|--------|"
    for _ in protocol_names:
        result += "-" * (len("Protocol Instrument - ") + 15) + "|"
    result += "\n"
    
    # Process results and build table rows
    for domain_result in results:
        domain = domain_result["domain"]
        instruments = domain_result.get("instruments", {})
        
        # Debug output
        print(f"Domain: {domain}, Instruments: {instruments}")
        
        result += f"| {domain} |"
        for protocol_name in protocol_names:
            instrument = instruments.get(protocol_name, "Not identified")
            # Clean up any trailing periods or whitespace
            if isinstance(instrument, str):
                instrument = instrument.strip().rstrip('.')
            result += f" {instrument} |"
        result += "\n"
    
    # Export to CSV if requested
    csv_path = None
    if export_csv:
        # Create output directory if it doesn't exist
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path for the CSV file
        filename = "domain_analysis.csv"
        csv_path = os.path.join(output_dir, filename)
        
        # Write the data to CSV
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                # Create fieldnames with all protocol names
                fieldnames = ['Domain']
                for protocol_name in protocol_names:
                    fieldnames.append(f'Protocol Instrument - {protocol_name}')
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for domain_result in results:
                    domain = domain_result["domain"]
                    instruments = domain_result.get("instruments", {})
                    
                    row = {'Domain': domain}
                    for protocol_name in protocol_names:
                        instrument = instruments.get(protocol_name, "Not identified")
                        if isinstance(instrument, str):
                            instrument = instrument.strip().rstrip('.')
                        row[f'Protocol Instrument - {protocol_name}'] = instrument
                    writer.writerow(row)
            
            # Store the CSV path in the user session
            cl.user_session.set("csv_path", csv_path)
            
        except Exception as e:
            result += f"\n\nError creating CSV file: {str(e)}"
    
    return result

async def _search_protocol_for_instruments(domain: str) -> dict:
    """Search the protocol for instruments related to a specific NIH HEAL CDE core domain."""
    global embedding_model
    
    # Get the session-specific Qdrant client
    session_qdrant_client = cl.user_session.get("session_qdrant_client")
    if not session_qdrant_client:
        return {"domain": domain, "instrument": "No session-specific Qdrant client found", "context": ""}
    
    # Get the names of the uploaded protocol files
    protocol_names = cl.user_session.get("protocol_filenames", ["Unknown Protocol"])
    
    # Check if user collection exists
    try:
        # Check if collection exists
        if USER_EMBEDDINGS_NAME not in [c.name for c in session_qdrant_client.get_collections().collections]:
            return {"domain": domain, "instrument": "No protocol document embedded", "context": ""}
            
        # Create retriever for user documents
        user_retriever = QdrantVectorStore(
            client=session_qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=embedding_model
        ).as_retriever(search_kwargs={"k": 20})  # Increase k to get more documents
    except Exception as e:
        print(f"Error accessing user vector store: {str(e)}")
        return {"domain": domain, "instrument": "Error accessing protocol", "context": str(e)}
    
    # Create the chat model with the specified model from constants
    domain_chat_model = ChatOpenAI(model_name=general_llm_model, temperature=0)
    
    # Results for each protocol
    protocol_results = {}
    
    # Search for instruments in the core reference data that match this domain
    core_reference_query = f"What are standard instruments or measures for {domain}?"
    core_reference_instruments = await core_reference_retrieval_chain.ainvoke({"question": core_reference_query})
    
    # Process each protocol separately
    for protocol_name in protocol_names:
        try:
            # Search for instruments related to this domain in this specific protocol
            query = f"What instrument or measure is used for {domain} in the protocol named {protocol_name}?"
            
            # Retrieve relevant chunks from the protocol
            docs = await user_retriever.ainvoke(query)
            
            # Filter documents to only include those from this protocol
            protocol_docs = [doc for doc in docs if doc.metadata.get("source") == protocol_name]
            
            # If no documents match this protocol, try a more general search
            if not protocol_docs:
                print(f"No specific docs found for {protocol_name}, using all retrieved docs")
                protocol_docs = docs
            
            protocol_context = format_docs(protocol_docs)
            
            # Use the model to identify the most likely instrument for this domain in this protocol
            prompt = f"""
            Based on the protocol information and known instruments, identify which instrument is being used for the domain: {domain} in the protocol: {protocol_name}
            
            Protocol information:
            {protocol_context}
            
            Known instruments for this domain:
            {core_reference_instruments}
            
            Respond with only the name of the identified instrument. If you cannot identify a specific instrument, respond with "Not identified".
            """
            
            instrument = await domain_chat_model.ainvoke([HumanMessage(content=prompt)])
            
            # Store the result for this protocol
            protocol_results[protocol_name] = {
                "instrument": instrument.content.strip(),
                "context": protocol_context
            }
            
            print(f"For {domain} in {protocol_name}: {instrument.content.strip()}")
            
        except Exception as e:
            print(f"Error identifying instrument for {domain} in {protocol_name}: {str(e)}")
            protocol_results[protocol_name] = {
                "instrument": "Error during identification",
                "context": str(e)
            }
    
    # Combine results into a single response
    combined_instruments = {}
    for protocol_name, result in protocol_results.items():
        combined_instruments[protocol_name] = result["instrument"]
    
    return {
        "domain": domain,
        "instruments": combined_instruments,
        "known_instruments": core_reference_instruments
    }

async def create_rag_chain(doc_type=None):
    """Create a RAG chain based on the document type."""
    # Get the session-specific Qdrant client
    session_qdrant_client = cl.user_session.get("session_qdrant_client")
    
    # Create retrievers based on document type
    if doc_type == "protocol" and session_qdrant_client:
        # Check if user collection exists
        try:
            if USER_EMBEDDINGS_NAME in [c.name for c in session_qdrant_client.get_collections().collections]:
                protocol_vectorstore = QdrantVectorStore(
                    client=session_qdrant_client,
                    collection_name=USER_EMBEDDINGS_NAME,
                    embedding=embedding_model
                )
                retriever = protocol_vectorstore.as_retriever(search_kwargs={"k": 5})
            else:
                raise ValueError("No protocol document embedded")
        except Exception as e:
            raise ValueError(f"Error accessing protocol: {str(e)}")
    elif doc_type == "core_reference":
        if core_vectorstore:
            retriever = core_vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            raise ValueError("Core reference data not available")
    else:
        # Default: search both if available
        retrievers = []
        
        # Add core reference retriever if available
        if core_vectorstore:
            core_retriever = core_vectorstore.as_retriever(search_kwargs={"k": 3})
            retrievers.append(core_retriever)
        
        # Add protocol retriever if available
        if session_qdrant_client:
            try:
                if USER_EMBEDDINGS_NAME in [c.name for c in session_qdrant_client.get_collections().collections]:
                    protocol_vectorstore = QdrantVectorStore(
                        client=session_qdrant_client,
                        collection_name=USER_EMBEDDINGS_NAME,
                        embedding=embedding_model
                    )
                    protocol_retriever = protocol_vectorstore.as_retriever(search_kwargs={"k": 3})
                    retrievers.append(protocol_retriever)
            except Exception as e:
                print(f"Error accessing protocol: {str(e)}")
        
        if not retrievers:
            raise ValueError("No data sources available")
        
        # If we have multiple retrievers, use them in sequence
        if len(retrievers) > 1:
            from langchain.retrievers import EnsembleRetriever
            retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=[1.0/len(retrievers)] * len(retrievers)
            )
        else:
            retriever = retrievers[0]
    
    # Create and return the RAG chain with async support
    return (
        {"context": itemgetter("question") | retriever | format_docs, 
         "question": itemgetter("question")}
        | rag_prompt 
        | chat_model
        | StrOutputParser()
    )

# Collect all tools - now just the two core tools
tools = [
    search_all_data,
    analyze_protocol_domains
]

# ==================== LANGGRAPH SETUP ====================
# LangGraph components
model = ChatOpenAI(model_name=general_llm_model, temperature=0)

# System message
system_message = """You are a helpful assistant specializing in NIH HEAL CDE protocols.

You have access to:
1. A tool to search all available data (search_all_data) - Use this to answer questions about the protocol or core reference data
2. A tool to analyze all NIH HEAL domains at once (analyze_protocol_domains) - This will identify instruments for each NIH HEAL CDE core domain, return the result in markdown, and also create a CSV file

WHEN TO USE TOOLS:
- When users ask general questions about the protocol or core reference data, use the search_all_data tool.
- When users want a complete analysis of all domains, use the analyze_protocol_domains tool.

Be specific in your tool queries to get the most relevant information.
Always use the appropriate tool before responding to questions about the protocol or core reference data.

IMPORTANT: When returning tool outputs, especially markdown tables or formatted content, preserve the exact formatting without adding any commentary, introduction, or conclusion.
"""

# Bind tools and configure models
model = model.bind_tools(tools)
tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we end the graph (reply to the user)
    return END

async def call_model(state: MessagesState):
    messages = state["messages"]
    # Add the system message at the beginning of the messages list
    if messages and not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Build the graph
builder = StateGraph(MessagesState)

builder.add_node("supervisor", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    should_continue,
)

builder.add_edge("tools", "supervisor")

graph = builder.compile()

# ==================== CHAINLIT HANDLERS ====================
@cl.on_chat_start
async def on_chat_start():
    # Create a session-specific Qdrant client
    session_qdrant_client = create_session_qdrant_client()
    cl.user_session.set("session_qdrant_client", session_qdrant_client)
    
    files = await cl.AskFileMessage(
        content="Please upload one or more NIH HEAL Protocols (PDFs only) for analysis.",
        accept=["application/pdf"],
        max_files=5,  # Allow up to 5 files
        max_size_mb=20,
        timeout=180,
    ).send()
    
    if not files:
        await cl.Message("Error: You must upload at least one PDF file.").send()
        return    
    
    if files:
        # Store the filenames in the user session
        protocol_filenames = [file.name for file in files]
        cl.user_session.set("protocol_filenames", protocol_filenames)
        
        processing_msg = cl.Message(content=f"Processing {len(files)} protocol(s)...")
        await processing_msg.send()
        
        # Process the uploaded files with the session-specific client
        documents_with_metadata = await load_and_chunk_protocol_files(files)
        user_vectorstore = await embed_protocol_in_qdrant(documents_with_metadata, session_qdrant_client)
        
        if user_vectorstore:
            # Present options to the user instead of automatically running analysis
            options_message = f"""
Your protocol(s) have been successfully processed! What would you like to do next?

1. Ask questions about the uploaded protocol(s)

2. Run a complete analysis of what core domain instruments are used in the uploaded protocol(s)
This will identify instruments for each NIH HEAL CDE core domain, return the result to your screen and create a downloadable crosswalk.
            """
            
            await cl.Message(content=options_message).send()
        else:
            await cl.Message(content="There was an issue processing your PDF(s). Please try uploading again.").send()

@cl.on_message
async def on_message(msg: cl.Message):
    # Show a thinking indicator
    with cl.Step("Heal SYNC to process your request"):
        final_answer = await process_message(msg.content)
    
    await final_answer.send()
    await handle_file_attachments()

async def process_message(content: str):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="")
    
    try:
        async for msg_response, metadata in graph.astream(
            {"messages": [HumanMessage(content=content)]}, 
            stream_mode="messages", 
            config=config
        ):
            # Process response
            if should_stream_response(msg_response, metadata):
                await final_answer.stream_token(msg_response.content)
    except Exception as e:
        # Handle graph processing errors gracefully
        await final_answer.stream_token(f"\n\nI encountered an error: {str(e)}")
    
    return final_answer

def should_stream_response(msg_response, metadata):
    return (
        msg_response.content
        and not isinstance(msg_response, HumanMessage)
        and metadata["langgraph_node"] == "supervisor"
    )

async def handle_file_attachments():
    csv_path = cl.user_session.get("csv_path")
    if not csv_path:
        return
        
    try:
        file_message = cl.Message(content="Download the crosswalk here:")
        await file_message.send()
        
        await cl.File(
            name="domain_analysis.csv",
            path=csv_path,
            display="inline"
        ).send(for_id=file_message.id)
        
        # Clear the path to avoid sending it multiple times
        cl.user_session.set("csv_path", None)
    except Exception as e:
        print(f"Error attaching CSV file: {str(e)}")