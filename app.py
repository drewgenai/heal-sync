import os
import shutil
import json
import pandas as pd
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.schema import HumanMessage
from typing_extensions import List, TypedDict
from operator import itemgetter
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()


UPLOAD_PATH = "upload/"
OUTPUT_PATH = "output/"
INITIAL_DATA_PATH = "./data/Instruments_Definitions.xlsx"
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize embeddings model
model_id = "Snowflake/snowflake-arctic-embed-m"
embedding_model = HuggingFaceEmbeddings(model_name=model_id)
semantic_splitter = SemanticChunker(embedding_model, add_start_index=True, buffer_size=30)
llm = ChatOpenAI(model="gpt-4o-mini")

# Export comparison prompt
export_prompt = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question.

Between these two files containing protocols, identify and match **entire assessment sections** based on conceptual similarity. Do NOT match individual questions.

### **Output Format:**
Return the response in **valid JSON format** structured as a list of dictionaries, where each dictionary contains:
[
    {{
        "Derived Description": "A short name for the matched concept",
        "Protocol_1": "Protocol 1 - Matching Element",
        "Protocol_2": "Protocol 2 - Matching Element"
    }},
    ...
]
### **Example Output:**
[
    {{
        "Derived Description": "Pain Coping Strategies",
        "Protocol_1": "Pain Coping Strategy Scale (PCSS-9)",
        "Protocol_2": "Chronic Pain Adjustment Index (CPAI-10)"
    }},
    {{
        "Derived Description": "Work Stress and Fatigue",
        "Protocol_1": "Work-Related Stress Scale (WRSS-8)",
        "Protocol_2": "Occupational Fatigue Index (OFI-7)"
    }},
    ...
]

### Rules:
1. Only output **valid JSON** with no explanations, summaries, or markdown formatting.
2. Ensure each entry in the JSON list represents a single matched data element from the two protocols.
3. If no matching element is found in a protocol, leave it empty ("").
4. **Do NOT include headers, explanations, or additional formatting**—only return the raw JSON list.
5. It should include all the elements in the two protocols.
6. If it cannot match the element, create the row and include the protocol it did find and put "could not match" in the other protocol column.
7. protocol should be the between
"""

compare_export_prompt = ChatPromptTemplate.from_template(export_prompt)

QUERY_PROMPT = """
You are a helpful assistant. Use the available context to answer the question concisely and informatively.

CONTEXT:
{context}

QUERY:
{question}

Provide a natural-language response using the given information. If you do not know the answer, say so.
"""

query_prompt = ChatPromptTemplate.from_template(QUERY_PROMPT)


@tool
def document_query_tool(question: str) -> str:
    """Retrieves relevant document sections and answers questions based on the uploaded documents."""

    retriever = cl.user_session.get("qdrant_retriever")
    if not retriever:
        return "Error: No documents available for retrieval. Please upload two PDF files first."
    retriever = retriever.with_config({"k": 10})

    # Use a RAG chain similar to the comparison tool
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | query_prompt | llm | StrOutputParser()
    )
    response_text = rag_chain.invoke({"question": question})

    # Get the retrieved docs for context
    retrieved_docs = retriever.invoke(question)

    return {
        "messages": [HumanMessage(content=response_text)],
        "context": retrieved_docs
    }


@tool
def document_comparison_tool(question: str) -> str:
    """Compares the two uploaded documents, identifies matched elements, exports them as JSON, formats into CSV, and provides a download link."""

    # Retrieve the vector database retriever
    retriever = cl.user_session.get("qdrant_retriever")
    if not retriever:
        return "Error: No documents available for retrieval. Please upload two PDF files first."

    # Process query using RAG
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | compare_export_prompt | llm | StrOutputParser()
    )
    response_text = rag_chain.invoke({"question": question})

    # Parse response and save as CSV
    try:
        structured_data = json.loads(response_text)
        if not structured_data:
            return "Error: No matched elements found."

        # Define output file path
        file_path = os.path.join(OUTPUT_PATH, "comparison_results.csv")

        # Save to CSV
        df = pd.DataFrame(structured_data, columns=["Derived Description", "Protocol_1", "Protocol_2"])
        df.to_csv(file_path, index=False)

        # Send the message with the file directly from the tool
        cl.run_sync(
            cl.Message(
                content="Comparison complete! Download the CSV below:",
                elements=[cl.File(name="comparison_results.csv", path=file_path, display="inline")],
            ).send()
        )
        
        # Return a simple confirmation message
        return "Comparison results have been generated and displayed."

    except json.JSONDecodeError:
        return "Error: Response is not valid JSON."


# Define tools for the agent
tools = [document_query_tool, document_comparison_tool]

# Set up the agent with a system prompt
system_prompt = """You are an intelligent document analysis assistant. You have access to two tools:

1. document_query_tool: Use this when a user wants information or has questions about the content of uploaded documents.
2. document_comparison_tool: Use this when a user wants to compare elements between two uploaded documents or export comparison results.

Analyze the user's request carefully to determine which tool is most appropriate.
"""

# Create the agent using OpenAI function calling
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    prompt=agent_prompt
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


def initialize_vector_store():
    """Initialize an empty Qdrant vector store"""
    try:
        # Create a Qdrant client for in-memory storage
        client = QdrantClient(location=":memory:")
        
        # Create the collection with the appropriate vector size
        # Snowflake/snowflake-arctic-embed-m produces 768-dimensional vectors
        vector_size = 768  # Changed from 1536 to match your embedding model
        
        # Check if collection exists, if not create it
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if "document_comparison" not in collection_names:
            client.create_collection(
                collection_name="document_comparison",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print("Created new collection: document_comparison")
        
        # Create the vector store with the client
        vectorstore = Qdrant(
            client=client,
            collection_name="document_comparison",
            embeddings=embedding_model
        )
        print("Vector store initialized successfully")
        return vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return None


async def load_reference_data(vectorstore):
    """Load reference Excel data into the vector database"""
    if not os.path.exists(INITIAL_DATA_PATH):
        print(f"Warning: Initial data file {INITIAL_DATA_PATH} not found")
        return vectorstore
    
    try:
        # Load Excel file
        df = pd.read_excel(INITIAL_DATA_PATH)
        
        # Convert DataFrame to documents
        documents = []
        for _, row in df.iterrows():
            # Combine all columns into a single text
            content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            doc = Document(page_content=content, metadata={"source": "Instruments_Definitions.xlsx"})
            documents.append(doc)
        
        # Add documents to vector store
        if documents:
            vectorstore.add_documents(documents)
            print(f"Successfully loaded {len(documents)} entries from {INITIAL_DATA_PATH}")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading reference data: {str(e)}")
        return vectorstore


async def process_uploaded_files(files, vectorstore):
    """Process uploaded PDF files and add them to the vector store"""
    documents_with_metadata = []
    for file in files:
        file_path = os.path.join(UPLOAD_PATH, file.name)
        shutil.copyfile(file.path, file_path)
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            source_name = file.name
            chunks = semantic_splitter.split_text(doc.page_content)
            for chunk in chunks:
                doc_chunk = Document(page_content=chunk, metadata={"source": source_name})
                documents_with_metadata.append(doc_chunk)
    
    if documents_with_metadata:
        # Add documents to vector store
        vectorstore.add_documents(documents_with_metadata)
        print(f"Added {len(documents_with_metadata)} chunks from uploaded files")
        return True
    return False


@cl.on_chat_start
async def start():
    # Initialize chat history for the agent
    cl.user_session.set("chat_history", [])
    
    # Initialize vector store
    vectorstore = initialize_vector_store()
    if not vectorstore:
        await cl.Message("Error: Could not initialize vector store.").send()
        return
    
    # Load reference data
    with cl.Step("Loading reference data"):
        vectorstore = await load_reference_data(vectorstore)
        cl.user_session.set("qdrant_vectorstore", vectorstore)
        cl.user_session.set("qdrant_retriever", vectorstore.as_retriever())
        await cl.Message("Reference data loaded successfully!").send()
    
    # Ask for PDF uploads
    files = await cl.AskFileMessage(
        content="Please upload **two PDF files** for comparison:",
        accept=["application/pdf"],
        max_files=2
    ).send()
    
    if len(files) != 2:
        await cl.Message("Error: You must upload exactly two PDF files.").send()
        return
    
    # Process uploaded files
    with cl.Step("Processing uploaded files"):
        success = await process_uploaded_files(files, vectorstore)
        if success:
            # Update the retriever with the latest vector store
            cl.user_session.set("qdrant_retriever", vectorstore.as_retriever())
            await cl.Message("Files uploaded and processed successfully! You can now enter your query.").send()
        else:
            await cl.Message("Error: Unable to process files. Please try again.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # Get chat history
    chat_history = cl.user_session.get("chat_history", [])
    
    # Run the agent
    with cl.Step("Agent thinking"):
        response = await cl.make_async(agent_executor.invoke)(
            {"input": message.content, "chat_history": chat_history}
        )
    
    # Handle the response based on the tool that was called
    if isinstance(response["output"], dict) and "messages" in response["output"]:
        # This is from document_query_tool
        await cl.Message(response["output"]["messages"][0].content).send()
    else:
        # Generic response (including the confirmation from document_comparison_tool)
        await cl.Message(content=str(response["output"])).send()
    
    # Update chat history with the new exchange
    chat_history.extend([
        HumanMessage(content=message.content),
        HumanMessage(content=str(response["output"]))
    ])
    cl.user_session.set("chat_history", chat_history)