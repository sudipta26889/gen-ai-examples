# imports
import streamlit as st
import os, tempfile
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Langsmith integration
import langsmith
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable Langsmith tracing (CORRECT format from documentation)
langsmith_key = os.getenv('LANGSMITH_API_KEY')
if langsmith_key:
    langsmith_client = langsmith.Client(api_key=langsmith_key)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    os.environ["LANGSMITH_PROJECT"] = "Document App"
    # Set for backward compatibility
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Document App"

# Pydantic models for structured responses
class DocumentChatResponse(BaseModel):
    """Structured response for Document queries"""
    answer: str = Field(description="The main answer to the user's question")
    confidence: float = Field(description="Confidence score from 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of how the answer was derived")
    data_used: List[str] = Field(description="Key data points used from the Document", default_factory=list)

# MUST be the first Streamlit command
st.set_page_config(page_title="Document AI", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def home_page():
    st.write("""Select any one feature from above sliderbox: \n
    1. Chat with PDF/CSV \n
    2. Summarize PDF/CSV  """)

@st.cache_resource()
def get_embeddings_model():
    """Initialize and cache the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_document(uploaded_file, *, chunk_size=1000, chunk_overlap=200):
    """Load an uploaded PDF/CSV into LangChain Documents and chunk them."""
    if not uploaded_file:
        raise ValueError("No file provided for ingestion.")

    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    file_extension = uploaded_file.name.split('.')[-1].lower()
    documents = None

    try:
        if file_extension == "pdf":
            loader = PyPDFLoader(file_path=tmp_path)
            documents = loader.load()
        elif file_extension == "csv":
            last_error = None
            for encoding in ("utf-8", "cp1252"):
                try:
                    loader = CSVLoader(file_path=tmp_path, encoding=encoding)
                    documents = loader.load()
                    break
                except Exception as csv_error:
                    last_error = csv_error
            if documents is None:
                raise last_error or ValueError("Unable to load CSV file with supported encodings.")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Please upload PDF or CSV files.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        return {
            "extension": file_extension,
            "documents": documents,
            "chunks": chunks,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def get_document_resources(uploaded_file, *, chunk_size=1000, chunk_overlap=200, require_vectorstore=False):
    """Cache document chunks (and optionally vectorstores) keyed by file metadata."""
    if not uploaded_file:
        raise ValueError("No file provided for ingestion.")

    cache = st.session_state.setdefault("document_cache", {})
    cache_key = (
        uploaded_file.name,
        getattr(uploaded_file, "size", None),
        chunk_size,
        chunk_overlap,
    )

    resources = cache.get(cache_key)
    if not resources:
        bundle = load_document(uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        resources = {"bundle": bundle}
        cache[cache_key] = resources

    if require_vectorstore and "vectorstore" not in resources:
        embeddings = get_embeddings_model()
        resources["vectorstore"] = FAISS.from_documents(
            documents=resources["bundle"]["chunks"],
            embedding=embeddings,
        )

    return resources

def retriever_func(uploaded_file):
    resources = get_document_resources(
        uploaded_file,
        chunk_size=1000,
        chunk_overlap=200,
        require_vectorstore=True,
    )
    vectorstore = resources["vectorstore"]
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever, vectorstore


def chat(temperature, model_name, user_api_key):
    st.write("# Talk to CSV/PDF")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your PDF or CSV here 👇:",
        type=["pdf", "csv"],
    )

    if not uploaded_file:
        st.info("Please upload a PDF or CSV file to start chatting.")
        return

    # Check if API key is valid
    if not user_api_key or user_api_key == "":
        st.error("❌ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return

    current_file_id = (uploaded_file.name, getattr(uploaded_file, "size", None))
    if st.session_state.get("active_chat_file") != current_file_id:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state["chat_histories"] = {}
        st.session_state["active_chat_file"] = current_file_id

    try:
        retriever, vectorstore = retriever_func(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return
    
    # Configure LLM for OpenRouter with explicit API key
    try:
        # Reduce max_tokens to fit within credit limit
        max_tokens = 1000  # Reduced from default to avoid credit issues

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            api_key=user_api_key,
            default_headers={
                "HTTP-Referer": "https://localhost:8501",
                "X-Title": "Document AI App"
            }
        )
        if "grok" in model_name.lower():
            st.success(f"🤖 Connected to {model_name} (max_tokens: {max_tokens}) - FREE MODEL! 🎉")
        else:
            st.success(f"🤖 Connected to {model_name} (max_tokens: {max_tokens})")
    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg and "credits" in error_msg:
            st.error("❌ **Insufficient Credits!**")
            st.error("💳 You need more OpenRouter credits to use this model.")
            st.error("🔗 Visit: https://openrouter.ai/settings/credits")
            st.warning("💡 Try using a smaller model like 'mistralai/mistral-7b-instruct'")
        else:
            st.error(f"❌ Failed to connect to OpenRouter: {error_msg}")
        return
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    store = st.session_state.setdefault("chat_histories", {})

    # Enhanced prompt with structured response format
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Use the following pieces of context to answer the question at the end.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer.

                  Context: {context}

                  Please provide a structured response in the following JSON format:
                  {{
                    "answer": "Your main answer here",
                    "confidence": 0.8,
                    "reasoning": "Brief explanation of how you arrived at the answer",
                    "data_used": ["key data points from context"]
                  }}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Use Pydantic parser for structured responses
    parser = PydanticOutputParser(pydantic_object=DocumentChatResponse)
    runnable = prompt | llm | parser
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle chat input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            # Get context from vectorstore
            context_results = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in context_results)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Use streaming response
                try:
                    # Get structured response from model
                    response = with_message_history.invoke(
                        {"context": context, "input": prompt},
                        config={"configurable": {"session_id": "abc123"}}
                    )

                    if isinstance(response, DocumentChatResponse):
                        # Format structured response nicely
                        full_response = f"""
**Answer:** {response.answer}

**Confidence:** {response.confidence:.2f}

**Reasoning:** {response.reasoning}
"""
                        if response.data_used:
                            full_response += f"\n**Data Used:**\n" + "\n".join(f"• {item}" for item in response.data_used)
                    elif hasattr(response, 'content'):
                        full_response = response.content
                    else:
                        full_response = str(response)

                    # Display the response
                    message_placeholder.markdown(full_response)

                    # Show raw JSON in expander for debugging
                    with st.expander("🔍 Raw Response (Debug)"):
                        if isinstance(response, DocumentChatResponse):
                            st.json({
                                "answer": response.answer,
                                "confidence": response.confidence,
                                "reasoning": response.reasoning,
                                "data_used": response.data_used
                            })
                        else:
                            st.text(str(response))

                except Exception as stream_error:
                    st.error(f"❌ Response parsing error: {stream_error}")
                    # Fallback to basic response
                    try:
                        response = with_message_history.invoke(
                            {"context": context, "input": prompt},
                            config={"configurable": {"session_id": "abc123"}}
                        )
                        full_response = response.content if hasattr(response, 'content') else str(response)
                        message_placeholder.markdown(f"**Basic Response:**\n\n{full_response}")
                    except Exception as fallback_error:
                        st.error(f"❌ Complete failure: {fallback_error}")
                        full_response = "Sorry, I encountered an error processing your request. Please try rephrasing your question."

                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})

    # Handle reset button
    if reset:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.rerun()

def summary(model_name, temperature, user_api_key):
    st.write("# Summarize PDF/CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader(
        "Upload source document",
        type=["pdf", "csv"],
        label_visibility="collapsed",
    )

    if not user_api_key or user_api_key == "":
        st.error("❌ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return

    if not uploaded_file:
        st.info("Please upload a PDF or CSV file to generate a summary.")
        return

    try:
        resources = get_document_resources(
            uploaded_file,
            chunk_size=1024,
            chunk_overlap=100,
            require_vectorstore=False,
        )
        doc_bundle = resources["bundle"]
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return

    chunks = doc_bundle["chunks"]
    file_label = doc_bundle["extension"].upper()

    if not chunks:
        st.warning("The uploaded document did not contain any readable text. Please try another file.")
        return

    st.info(f"📄 Loaded {len(chunks)} text chunks from your {file_label} file")

    if st.button("Generate Summary"):
        with st.spinner("Generating summary... This may take a moment."):
            try:
                max_tokens = 1000

                llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=user_api_key,
                    default_headers={
                        "HTTP-Referer": "https://localhost:8501",
                        "X-Title": "Document AI App"
                    }
                )
                st.success(f"🤖 Connected to {model_name} for summarization (max_tokens: {max_tokens})")
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    return_intermediate_steps=True,
                    input_key="input_documents",
                    output_key="output_text",
                )
                result = chain({"input_documents": chunks}, return_only_outputs=True)
                st.success("✅ Summary generated successfully!")
                st.markdown("### 📋 Summary:")
                st.write(result["output_text"])
            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")
                st.info("💡 Try using a smaller document or check your API key.")


# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 Document AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Chat with and Summarize Your PDFs & CSVs!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Check for OpenRouter API key
    if os.path.exists(".env") and os.environ.get("OPENROUTER_API_KEY"):
        user_api_key = os.environ["OPENROUTER_API_KEY"]
        st.success("OpenRouter API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter OpenRouter API key 👇", 
            placeholder="Paste your OpenRouter API key, sk-or-v1-...", 
            type="password", 
            key="openrouter_api_key"
        )
        if user_api_key:
            st.sidebar.success("OpenRouter API key loaded", icon="🚀")

    # OpenRouter model options
    MODEL_OPTIONS = [
        "x-ai/grok-4-fast:free",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-haiku-20240307",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-pro-1.5-latest",
        "mistralai/mistral-7b-instruct"
    ]
    
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.9
    TEMPERATURE_STEP = 0.01
    
    model_name = st.sidebar.selectbox(
        label="Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("x-ai/grok-4-fast:free")  # Set Grok as default
    )
    temperature = st.sidebar.slider(
                label="Temperature",
                min_value=TEMPERATURE_MIN_VALUE,
                max_value=TEMPERATURE_MAX_VALUE,
                value=TEMPERATURE_DEFAULT_VALUE,
                step=TEMPERATURE_STEP,)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Setup**:\n"
        "- **OpenRouter API**: For chat models ([openrouter.ai](https://openrouter.ai))\n"
        "- **Embeddings**: Free HuggingFace model (no API key needed!)\n\n"
        "🚀 **Using**: `sentence-transformers/all-MiniLM-L6-v2`\n"
        "🎯 **Free Model**: `x-ai/grok-4-fast:free` (default)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Dependencies**: Make sure you have:\n"
        "```\n"
        "pip install langchain-openai langchain-community\n"
        "pip install sentence-transformers\n"
        "pip install --upgrade pydantic\n"
        "```"
    )

    functions = [
        "home",
        "Chat",
        "Summarize",
    ]
    
    selected_function = st.selectbox("Select a functionality", functions)
    
    if selected_function != "home" and not user_api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to use this functionality.")
        return
    
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat":
        chat(temperature=temperature, model_name=model_name, user_api_key=user_api_key)
    elif selected_function == "Summarize":
        summary(temperature=temperature, model_name=model_name, user_api_key=user_api_key)
    else:
        st.warning("You haven't selected any AI Functionality!!")

if __name__ == "__main__":
    main()
