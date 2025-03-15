import os
import shutil
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
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

# Initialize session state if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qdrant_vectorstore" not in st.session_state:
    st.session_state.qdrant_vectorstore = None
if "qdrant_retriever" not in st.session_state:
    st.session_state.qdrant_retriever = None
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

UPLOAD_PATH = "upload/"
OUTPUT_PATH = "output/"
DATA_PATH = "./data/"
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

    retriever = st.session_state.qdrant_retriever
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
    retriever = st.session_state.qdrant_retriever
    if not retriever:
        return "Error: No documents available for retrieval. Please upload two PDF files first."
    
    # Set k=10 to match the document_query_tool
    retriever = retriever.with_config({"k": 10})

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

        # In Streamlit, we'll handle the file download in the main app flow
        st.session_state.comparison_results = file_path
        
        # Return a simple confirmation message
        return "Comparison results have been generated and are ready for download."

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
        
        # Snowflake/snowflake-arctic-embed-m produces 768-dimensional vectors
        vector_size = 768
        
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
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="document_comparison",
            embedding=embedding_model
        )
        print("Vector store initialized successfully")
        return vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return None


def load_reference_data(vectorstore):
    """Load all Excel files from the data directory into the vector database"""
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data directory {DATA_PATH} not found")
        return vectorstore
    
    try:
        # Get all Excel files in the data directory
        excel_files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print(f"Warning: No Excel files found in {DATA_PATH}")
            return vectorstore
            
        # Add a flag to track if we've already loaded these files
        if hasattr(vectorstore, '_reference_data_loaded'):
            print("Reference data already loaded, skipping...")
            return vectorstore
            
        total_documents = 0
        
        # Process each Excel file
        for file_name in excel_files:
            file_path = os.path.join(DATA_PATH, file_name)
            df = pd.read_excel(file_path)
            
            # Convert DataFrame to documents
            documents = []
            for _, row in df.iterrows():
                # Combine all columns into a single text
                content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
                doc = Document(page_content=content, metadata={"source": file_name})
                documents.append(doc)
            
            # Add documents to vector store
            if documents:
                vectorstore.add_documents(documents)
                total_documents += len(documents)
                print(f"Successfully loaded {len(documents)} entries from {file_name}")
        
        print(f"Total entries loaded: {total_documents} from {len(excel_files)} files")
        
        # Mark that we've loaded the reference data
        setattr(vectorstore, '_reference_data_loaded', True)
        
        return vectorstore
    except Exception as e:
        print(f"Error loading reference data: {str(e)}")
        return vectorstore


def process_uploaded_files(files, vectorstore):
    """Process uploaded PDF files and add them to the vector store"""
    print(f"Processing {len(files)} uploaded files")
    documents_with_metadata = []
    for file in files:
        print(f"Processing file: {file.name}, size: {file.size} bytes")
        file_path = os.path.join(UPLOAD_PATH, file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
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


# Streamlit UI
st.title("Document Analysis Assistant")

# Initialize vector store on first run
if st.session_state.qdrant_vectorstore is None:
    with st.spinner("Initializing vector store..."):
        vectorstore = initialize_vector_store()
        if vectorstore:
            st.session_state.qdrant_vectorstore = vectorstore
            vectorstore = load_reference_data(vectorstore)
            st.session_state.qdrant_retriever = vectorstore.as_retriever()
            st.success("Reference data loaded successfully!")
        else:
            st.error("Error: Could not initialize vector store.")

# File upload section
if not st.session_state.files_processed:
    st.write("Please upload two PDF files for comparison:")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) == 2:
        if st.button("Process Files"):
            with st.spinner("Processing uploaded files..."):
                success = process_uploaded_files(uploaded_files, st.session_state.qdrant_vectorstore)
                if success:
                    # Update the retriever with the latest vector store
                    st.session_state.qdrant_retriever = st.session_state.qdrant_vectorstore.as_retriever()
                    st.session_state.files_processed = True
                    st.success("Files uploaded and processed successfully! You can now enter your query.")
                    st.rerun()
                else:
                    st.error("Error: Unable to process files. Please try again.")
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning("Please upload exactly two PDF files for comparison.")

# Chat interface
if st.session_state.files_processed:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message)
                
                # If there are comparison results to download
                if "comparison_results" in st.session_state and isinstance(message, str) and "Comparison results have been generated" in message:
                    with open(st.session_state.comparison_results, "rb") as file:
                        st.download_button(
                            label="Download Comparison Results",
                            data=file,
                            file_name="comparison_results.csv",
                            mime="text/csv"
                        )
    
    # Input for new message
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Process with agent
        with st.spinner("Thinking..."):
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history}
            )
        
        # Display assistant response
        with st.chat_message("assistant"):
            if isinstance(response["output"], dict) and "messages" in response["output"]:
                # This is from document_query_tool
                st.write(response["output"]["messages"][0].content)
                # Store the response in chat history
                st.session_state.chat_history.append(response["output"]["messages"][0].content)
            else:
                # Generic response (including the confirmation from document_comparison_tool)
                st.write(response["output"])
                # Store the response in chat history
                st.session_state.chat_history.append(response["output"])
                
                # If there are comparison results to download
                if "comparison_results" in st.session_state and "Comparison results have been generated" in response["output"]:
                    with open(st.session_state.comparison_results, "rb") as file:
                        st.download_button(
                            label="Download Comparison Results",
                            data=file,
                            file_name="comparison_results.csv",
                            mime="text/csv"
                        )
    