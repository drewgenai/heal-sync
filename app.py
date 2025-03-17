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

# ==================== CONSTANTS ====================
# Paths and directories
UPLOAD_PATH = "./uploads"
INITIAL_EMBEDDINGS_DIR = "./initial_embeddings"
INITIAL_EMBEDDINGS_NAME = "initial_embeddings"
USER_EMBEDDINGS_NAME = "user_embeddings"
VECTOR_STORE_COLLECTION = "documents"

# Model IDs
EMBEDDING_MODEL_ID = "pritamdeka/S-PubMedBert-MS-MARCO"
#EMBEDDING_MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
INSTRUMENT_SEARCH_LLM = "gpt-4o"  # LLM for searching instruments
INSTRUMENT_ANALYSIS_LLM = "gpt-4o"  # LLM for analyzing all domains

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

# Make sure upload directory exists
os.makedirs(UPLOAD_PATH, exist_ok=True)

# ==================== EMBEDDING MODEL SETUP ====================
def get_embedding_model(model_id):
    """Creates and returns the appropriate embedding model based on the model ID."""
    if "text-embedding" in model_id:
        # OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_id)
    else:
        # HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name=model_id)

def initialize_embedding_models():
    """Initialize a single embedding model for all document types"""
    global embedding_model
    
    # Initialize a single model for all document types
    embedding_model = get_embedding_model(EMBEDDING_MODEL_ID)
    
    print(f"Initialized embedding model: {EMBEDDING_MODEL_ID}")

# Initialize the embedding model
initialize_embedding_models()

# Get embedding dimensions utility
def get_embedding_dimensions(model_id):
    """Gets the dimensions of embeddings from a specific model."""
    model = get_embedding_model(model_id)
    sample_text = "Sample text to determine embedding dimension"
    sample_embedding = model.embed_query(sample_text)
    return len(sample_embedding)

# ==================== QDRANT SETUP ====================
# Initialize Qdrant (in-memory)
qdrant_client = QdrantClient(":memory:")

# ==================== DOCUMENT PROCESSING ====================
# Create a semantic splitter for documents
semantic_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def format_docs(docs):
    """Format a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ==================== EXCEL DOCUMENT PROCESSING ====================
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
                    chunk.metadata["type"] = "excel"  # Add document type
                
                all_chunks.extend(chunks)
                file_count += 1
                print(f"Processed: {file} - {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    print(f"Processed {file_count} Excel files with a total of {len(all_chunks)} chunks.")
    return all_chunks

def embed_chunks_in_qdrant(chunks):
    """Embeds document chunks and stores them in Qdrant."""
    global embedding_model
    
    if not chunks:
        print("No Excel files found to process or all files were empty.")
        return None

    # Ensure we have a valid embedding model
    if embedding_model is None:
        print("ERROR: No embedding model available. Initializing now.")
        initialize_embedding_models()
        
    print(f"Using embedding model: {EMBEDDING_MODEL_ID}")
    print("Creating vector store...")
    
    try:
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            location=":memory:",
            collection_name=INITIAL_EMBEDDINGS_NAME
        )
        print(f"Successfully loaded all .xlsx files into Qdrant collection '{INITIAL_EMBEDDINGS_NAME}'.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        print(f"Embedding model status: {embedding_model is not None}")
        return None

def process_initial_embeddings():
    """Loads all .xlsx files, extracts text, embeds, and stores in Qdrant."""
    chunks = load_and_chunk_excel_files()
    return embed_chunks_in_qdrant(chunks)

# ==================== PDF DOCUMENT PROCESSING ====================
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
                    doc_chunk = Document(
                        page_content=chunk, 
                        metadata={
                            "source": source_name,
                            "type": "pdf"  # Add document type
                        }
                    )
                    documents_with_metadata.append(doc_chunk)
                    
            print(f"Successfully processed {file.name}, extracted {len(documents_with_metadata)} chunks")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    return documents_with_metadata

async def embed_pdf_chunks_in_qdrant(documents_with_metadata, model_name=EMBEDDING_MODEL_ID):
    """Create a vector store and embed PDF chunks into Qdrant."""
    global embedding_model
    
    if not documents_with_metadata:
        print("No documents to embed")
        return None
        
    print(f"Using embedding model: {model_name}")
    
    try:
        # First, check if collection exists and delete it if it does
        if USER_EMBEDDINGS_NAME in [c.name for c in qdrant_client.get_collections().collections]:
            qdrant_client.delete_collection(USER_EMBEDDINGS_NAME)
            
        # Create the collection with proper parameters
        # Get the embedding dimension from the model
        embedding_dimension = len(embedding_model.embed_query("Sample text"))
        
        qdrant_client.create_collection(
            collection_name=USER_EMBEDDINGS_NAME,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        
        # Create the vector store
        user_vectorstore = QdrantVectorStore(
            client=qdrant_client,
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

async def process_uploaded_files(files, model_name=EMBEDDING_MODEL_ID):
    """Process uploaded PDF files and add them to a separate vector store collection"""
    documents_with_metadata = await load_and_chunk_pdf_files(files)
    return await embed_pdf_chunks_in_qdrant(documents_with_metadata, model_name)


# ==================== RETRIEVAL FUNCTIONS ====================
def retrieve_documents(query, doc_type=None, k=5):
    """Retrieve documents, optionally filtering by document type"""
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=VECTOR_STORE_COLLECTION,
        embedding=embedding_model
    )
    
    # Set up filter if doc_type is specified
    search_kwargs = {"k": k}
    if doc_type:
        search_kwargs["filter"] = {"type": doc_type}
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)

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
chat_model = ChatOpenAI()

# Create a RAG chain that can be filtered by document type
def create_rag_chain(doc_type=None):
    """Create a RAG chain that can be filtered by document type"""
    def retrieve_with_type(query):
        docs = retrieve_documents(query, doc_type=doc_type)
        return format_docs(docs)
    
    chain = (
        {"context": lambda x: retrieve_with_type(x["question"]), 
         "question": itemgetter("question")}
        | rag_prompt 
        | chat_model
        | StrOutputParser()
    )
    
    return chain

# Initialize the Excel retriever
vectorstore = process_initial_embeddings()
if vectorstore:
    excel_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("Excel retriever created successfully.")
else:
    print("Failed to create Excel retriever: No vector store available.")

# Chain for retrieving from Excel embeddings
initialembeddings_retrieval_chain = (
    {"context": itemgetter("question") | excel_retriever | format_docs, 
     "question": itemgetter("question")}
    | rag_prompt 
    | chat_model
    | StrOutputParser()
)

# ==================== TOOL DEFINITIONS ====================
@tool
def search_data(query: str, doc_type: str = None) -> str:
    """Search all data or filter by document type (pdf/excel)"""
    try:
        chain = create_rag_chain(doc_type)
        return chain.invoke({"question": query})
    except Exception as e:
        return f"Error searching data: {str(e)}"

@tool
def search_excel_data(query: str, top_k: int = 3) -> str:
    """Search both Excel data and user-uploaded PDF data for information related to the query."""
    global embedding_model
    
    # Use the existing initialembeddings_retrieval_chain
    result = initialembeddings_retrieval_chain.invoke({"question": query})
    
    # If we have a user collection, also search that
    try:
        # Check if user collection exists
        if USER_EMBEDDINGS_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
            # If no user collection exists yet, just return Excel results
            return result
            
        # Create a retrieval chain for user documents
        user_retriever = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=embedding_model
        ).as_retriever(search_kwargs={"k": top_k})
        
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
        print(f"Error searching user vector store: {str(e)}")
        # If error occurs, just return Excel results
        return result

@tool
def load_and_embed_protocol_pdf(file_path: str = None) -> str:
    """Load and embed a protocol PDF file into the vector store.
    
    Args:
        file_path: Optional path to the PDF file. If None, will use files in the upload directory.
        
    Returns:
        String indicating success or failure of the embedding process
    """
    try:
        # If no specific file path is provided, use all PDFs in the upload directory
        if not file_path:
            uploaded_files = [f for f in os.listdir(UPLOAD_PATH) if f.endswith('.pdf')]
            if not uploaded_files:
                return "No protocol documents found in the upload directory."
            
            # Create file objects for processing
            files = []
            for filename in uploaded_files:
                file_path = os.path.join(UPLOAD_PATH, filename)
                # Create a simple object with the necessary attributes
                class FileObj:
                    def __init__(self, path, name, size):
                        self.path = path
                        self.name = name
                        self.size = size
                
                file_size = os.path.getsize(file_path)
                files.append(FileObj(file_path, filename, file_size))
        else:
            # Create a file object for the specific file
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            class FileObj:
                def __init__(self, path, name, size):
                    self.path = path
                    self.name = name
                    self.size = size
            
            files = [FileObj(file_path, filename, file_size)]
        
        # Process the files asynchronously
        import asyncio
        documents_with_metadata = asyncio.run(load_and_chunk_pdf_files(files))
        user_vectorstore = asyncio.run(embed_pdf_chunks_in_qdrant(documents_with_metadata, EMBEDDING_MODEL_ID))
        
        if user_vectorstore:
            return f"Successfully embedded {len(documents_with_metadata)} chunks from {len(files)} protocol document(s)."
        else:
            return "Failed to embed protocol document(s)."
    except Exception as e:
        return f"Error embedding protocol document: {str(e)}"

@tool
def search_protocol(query: str, top_k: int = 5) -> str:
    """Search the protocol for information related to the query."""
    global embedding_model
    
    try:
        # Check if user collection exists
        if USER_EMBEDDINGS_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
            return "No protocol document has been embedded yet. Please upload and embed a protocol first."
        
        # Create a retrieval chain for user documents
        user_retriever = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=embedding_model
        ).as_retriever(search_kwargs={"k": top_k})
        
        user_retrieval_chain = (
            {"context": itemgetter("question") | user_retriever | format_docs, 
             "question": itemgetter("question")}
            | rag_prompt 
            | chat_model
            | StrOutputParser()
        )
        
        result = user_retrieval_chain.invoke({"question": query})
        return result
    except Exception as e:
        return f"Error searching protocol: {str(e)}"

@tool
def search_protocol_for_instruments(domain: str) -> dict:
    """Search the protocol for instruments related to a specific NIH HEAL CDE core domain."""
    global embedding_model
    
    # Check if user collection exists
    try:
        # Check if collection exists
        if USER_EMBEDDINGS_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
            return {"domain": domain, "instrument": "No protocol document embedded", "context": ""}
            
        # Create retriever for user documents
        user_retriever = QdrantVectorStore(
            client=qdrant_client,
            collection_name=USER_EMBEDDINGS_NAME,
            embedding=embedding_model
        ).as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        print(f"Error accessing user vector store: {str(e)}")
        return {"domain": domain, "instrument": "Error accessing protocol", "context": str(e)}
    
    # Create the chat model with the specified model from constants
    domain_chat_model = ChatOpenAI(model_name=INSTRUMENT_SEARCH_LLM, temperature=0)
    
    # Search for instruments related to this domain in the protocol
    query = f"What instrument or measure is used for {domain} in the protocol?"
    
    try:
        # Retrieve relevant chunks from the protocol
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
        
        instrument = domain_chat_model.invoke([HumanMessage(content=prompt)]).content
        
        # Return the results as a dictionary
        return {
            "domain": domain,
            "instrument": instrument.strip(),
            "context": protocol_context,
            "known_instruments": excel_instruments
        }
    except Exception as e:
        print(f"Error identifying instrument for {domain}: {str(e)}")
        return {"domain": domain, "instrument": "Error during identification", "context": str(e)}

@tool
def analyze_domain(domain: str) -> dict:
    """Analyze a specific NIH HEAL CDE core domain"""
    # Query for this specific domain
    query = f"What instrument or measure is used for {domain} in the protocol?"
    
    # Get protocol context
    protocol_docs = retrieve_documents(query, doc_type="pdf", k=5)
    protocol_context = format_docs(protocol_docs)
    
    # Get known instruments from Excel data
    excel_query = f"What are standard instruments or measures for {domain}?"
    excel_docs = retrieve_documents(excel_query, doc_type="excel", k=5)
    excel_context = format_docs(excel_docs)
    
    # Use the model to identify the instrument
    prompt = f"""
    Based on the protocol information and known instruments, identify which instrument is being used for the domain: {domain}
    
    Protocol information:
    {protocol_context}
    
    Known instruments for this domain:
    {excel_context}
    
    Respond with only the name of the identified instrument. If you cannot identify a specific instrument, respond with "Not identified".
    """
    
    instrument = ChatOpenAI(model_name=INSTRUMENT_SEARCH_LLM, temperature=0).invoke(
        [HumanMessage(content=prompt)]
    ).content
    
    return {
        "domain": domain,
        "instrument": instrument.strip(),
        "context": protocol_context
    }

@tool
def analyze_all_heal_domains() -> str:
    """Analyze all NIH HEAL CDE core domains and identify instruments used in the protocol.
    
    Returns:
        Markdown formatted table of domains and identified instruments
    """
    # Check if protocol document exists
    uploaded_files = [f for f in os.listdir(UPLOAD_PATH) if f.endswith('.pdf')]
    if not uploaded_files:
        return "No protocol document has been uploaded yet."
    
    # For each domain, search for relevant instruments
    domain_instruments = {}
    
    for domain in NIH_HEAL_CORE_DOMAINS:
        # Use the search_protocol_for_instruments tool to get results for each domain
        result = search_protocol_for_instruments(domain)
        domain_instruments[domain] = result["instrument"]
        print(f"Identified instrument for {domain}: {result['instrument']}")
    
    # Format the results as a markdown table
    result = "# NIH HEAL CDE Core Domains and Identified Instruments\n\n"
    result += "| Domain | Protocol Instrument |\n"
    result += "|--------|--------------------|\n"
    
    for domain, instrument in domain_instruments.items():
        result += f"| {domain} | {instrument} |\n"
    
    return result

@tool
def analyze_all_domains() -> str:
    """Analyze all NIH HEAL CDE core domains at once"""
    results = []
    
    for domain in NIH_HEAL_CORE_DOMAINS:
        result = analyze_domain(domain)
        results.append(result)
    
    # Format as markdown table
    markdown = "# NIH HEAL CDE Core Domains Analysis\n\n"
    markdown += "| Domain | Protocol Instrument |\n"
    markdown += "|--------|--------------------|\n"
    
    for result in results:
        markdown += f"| {result['domain']} | {result['instrument']} |\n"
    
    return markdown

@tool
def format_instrument_analysis(analysis_results: list, title: str = "NIH HEAL CDE Core Domains Analysis") -> str:
    """Format instrument analysis results into a markdown table.
    
    Args:
        analysis_results: List of dictionaries with domain and instrument information
        title: Title for the markdown output
        
    Returns:
        Markdown formatted table of domains and identified instruments
    """
    # Format the results as a markdown table
    result = f"# {title}\n\n"
    result += "| Domain | Protocol Instrument |\n"
    result += "|--------|--------------------|\n"
    
    for item in analysis_results:
        domain = item.get("domain", "Unknown")
        instrument = item.get("instrument", "Not identified")
        result += f"| {domain} | {instrument} |\n"
    
    return result

# Collect all tools
tools = [
    search_data,
    search_excel_data,
    load_and_embed_protocol_pdf,
    search_protocol, 
    search_protocol_for_instruments, 
    analyze_domain,
    analyze_all_heal_domains,
    analyze_all_domains,
    format_instrument_analysis
]

# ==================== LANGGRAPH SETUP ====================
# LangGraph components
model = ChatOpenAI(model_name=INSTRUMENT_ANALYSIS_LLM, temperature=0)
final_model = ChatOpenAI(model_name=INSTRUMENT_ANALYSIS_LLM, temperature=0)

# System message
system_message = """You are a helpful assistant specializing in NIH HEAL CDE protocols.

You have access to:
1. Excel data through the search_excel_data tool
2. A tool to load and embed protocol PDFs (load_and_embed_protocol_pdf)
3. A tool to search protocol documents for general information (search_protocol)
4. A tool to search for instruments in protocols for specific domains (search_protocol_for_instruments)
5. A tool to analyze all NIH HEAL domains at once (analyze_all_heal_domains)
6. A tool to format analysis results into a markdown table (format_instrument_analysis)

WHEN TO USE TOOLS:
- When users upload a protocol PDF, use the load_and_embed_protocol_pdf tool.
- When users ask general questions about the protocol, use the search_protocol tool.
- When users ask about a specific instrument for a domain, use the search_protocol_for_instruments tool.
- When users want a complete analysis of all domains, use the analyze_all_heal_domains tool.
- When users ask about data or information in the Excel files, use the search_excel_data tool.
- When you have multiple analysis results to present, use format_instrument_analysis to create a nice table.

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

# ==================== CHAINLIT HANDLERS ====================
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
        documents_with_metadata = await load_and_chunk_pdf_files(files)
        user_vectorstore = await embed_pdf_chunks_in_qdrant(documents_with_metadata)
        
        if user_vectorstore:
            analysis_msg = cl.Message(content="Analyzing your protocol to identify instruments (CRF questionaires) for NIH HEAL CDE core domains...")
            await analysis_msg.send()
            
            # Use the analyze_all_heal_domains tool to analyze the protocol
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


