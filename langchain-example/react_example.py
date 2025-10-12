import argparse
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient
import wikipedia
import arxiv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# take input from cli
parser = argparse.ArgumentParser(description="LangChain ReAct example runner")
parser.add_argument("--query", help="Question to answer", default="who is sudipta?")
args = parser.parse_args()
query = args.query

print(f"Trying to find answer for the following Query: {query}")


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

# HuggingFace model for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Create a chat model
llm = ChatOpenAI(
    model="openai/gpt-4o", 
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1"
)

# Define a tool the agent can use
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query: str) -> str:
    """Search the web using Tavily and return formatted results with source tracking"""
    try:
        response = tavily_client.search(query)
        results = response.get("results", [])
        
        if not results:
            return "NO_RESULTS: No search results found."
        
        formatted_results = []
        sources_list = []
        urls = []
        
        for i, result in enumerate(results[:5], 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")
            formatted_results.append(f"{i}. {title}\n{content}\n")
            sources_list.append(f"- [{title}]({url})")
            urls.append(url)
        
        # Store source info globally
        
        result_text = "\n".join(formatted_results)
        sources_text = "\n".join(sources_list)
        
        return f"{result_text}\n\n**Web Sources:**\n{sources_text}"
        
    except Exception as e:
        return f"ERROR: Failed to search web - {str(e)}"

def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic information with fallback"""
    try:
        # Set language to English and limit results
        wikipedia.set_lang("en")
        
        # Search for articles
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            # Fallback: Try with simplified query (first few words)
            simplified_query = ' '.join(query.split()[:3])
            search_results = wikipedia.search(simplified_query, results=3)
            
            if not search_results:
                return "NO_RESULTS: No Wikipedia articles found."
        
        # Try to get the first article
        for article_name in search_results:
            try:
                page = wikipedia.page(article_name)
                summary = wikipedia.summary(article_name, sentences=4)
                
                # Store source info globally                
                result = f"**Wikipedia Article: {page.title}**\n\n"
                result += f"{summary}\n\n"
                
                # Add alternative articles if available
                if len(search_results) > 1:
                    other_articles = [a for a in search_results if a != article_name]
                    result += f"Related articles: {', '.join(other_articles[:2])}\n"
                
                result += f"\n**Wikipedia Source:** [{page.title}]({page.url})"
                return result
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages - try first option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=4)
                        
                        # Store source info globally                        
                        result = f"**Wikipedia Article: {page.title}** (from disambiguation)\n\n"
                        result += f"{summary}\n\n"
                        result += f"Other options: {', '.join(e.options[1:5])}\n"
                        result += f"\n**Wikipedia Source:** [{page.title}]({page.url})"
                        return result
                    except:
                        continue
                        
            except wikipedia.exceptions.PageError:
                continue
        
        return "NO_RESULTS: Wikipedia pages not accessible for this query."
            
    except Exception as e:
        return f"ERROR: Wikipedia search failed - {str(e)}"

def arxiv_search(query: str) -> str:
    """Search ArXiv for academic papers with improved fallback"""
    try:
        # Create search client
        client = arxiv.Client()
        
        # First attempt with original query
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results_list = list(client.results(search))
        
        # If no results, try with broader search terms
        if not results_list:
            # Remove special characters and try again
            simplified_query = ' '.join(query.replace('"', '').replace("'", '').split()[:5])
            search = arxiv.Search(
                query=simplified_query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results_list = list(client.results(search))
        
        if not results_list:
            return "NO_RESULTS: No academic papers found on ArXiv."
        
        formatted_results = []
        sources_list = []
        urls = []
        
        for i, result in enumerate(results_list, 1):
            paper_info = f"**{i}. {result.title}**\n"
            paper_info += f"Authors: {', '.join([author.name for author in result.authors[:3]])}\n"
            paper_info += f"Published: {result.published.strftime('%Y-%m-%d')}\n"
            paper_info += f"Abstract: {result.summary[:300]}...\n"
            formatted_results.append(paper_info)
            
            sources_list.append(f"- [{result.title}]({result.entry_id})")
            urls.append(result.entry_id)
        
        # Store source info globally        
        result_text = "\n".join(formatted_results)
        sources_text = "\n".join(sources_list)
        
        return f"{result_text}\n\n**ArXiv Papers:**\n{sources_text}"
        
    except Exception as e:
        return f"ERROR: ArXiv search failed - {str(e)}"

tools = [
    Tool(
        name="tavily_search",
        func=tavily_search,
        description="Search the web using Tavily and return formatted results with source tracking"
    ),
    Tool(
        name="wikipedia_search",
        func=wikipedia_search,
        description="Search Wikipedia for encyclopedic information with fallback"
    ),
    Tool(
        name="arxiv_search",
        func=arxiv_search,
        description="Search ArXiv for academic papers with improved fallback"
    )
]

# Load PDF
print("\nLoading Profile.pdf for RAG...\n")
loader = PyPDFLoader("./Profile.pdf")
pdf_doc = loader.load()

print("\nPDF Loaded. Now Chunking it...\n")

text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )

all_docs_texts = text_splitter.split_documents(pdf_doc)

print("\nPDF Chunked. Now Embedding it and storing it in FAISS VectorStore...\n")
# HuggingFace embeddings (force CPU load to avoid meta tensor issues)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)
db = FAISS.from_documents(all_docs_texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})


print("\nFirst RAG attempt, checking if answer of the query is in the context of the PDF or not...\n")
rag_prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the following context. 
                If the context doesn't contain enough information to answer the question completely,
                say "INSUFFICIENT_CONTEXT" at the beginning of your response.

                Context: {context}

                Question: {input}
                
                Answer:""")

document_chain = create_stuff_documents_chain(llm, rag_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

rag_result = retrieval_chain.invoke({"input": query})
rag_answer = rag_result["answer"].strip()

if "INSUFFICIENT_CONTEXT" not in rag_answer and len(rag_answer) > 20:
    print("\n!!! RAG found relevant info !!! \n")
    print(rag_answer)
else:
    print("\nXXX Didn't found info in RAG... XXX \n")
    print("\n Going to ask react agent to use tools to find the answer...\n")
    # Create a ReAct agent
    # prompt = hub.pull("hwchase17/react")
    prompt = ChatPromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    agent = create_react_agent(llm, tools, prompt)

    # Wrap in an executor to run
    executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    # Run agent
    result = executor.invoke({"input": query})
    print(result["output"])



