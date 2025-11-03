from langgraph.types import StreamWriter
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Sequence
import asyncio , os , re


class AgentState(TypedDict):
    message: Annotated[Sequence[dict], ...]

Model_name = "sqlCoder"

llm = ChatOllama(
    model=Model_name,
    temperature=0.4,
    num_predict=2048,
)
embeder = OllamaEmbeddings(model=Model_name)

def msg_frame(role: str, message: str) -> dict:
    match role:
        case "a":
            role = "Assistant"
        case "u":
            role = "User"
        case "s":
            role = "System"
        case "t":
            role = "Tool"
        case _:
            raise ValueError
    return {"role": str(role), "content": str(message)}

TOOLS_DESCRIPTION = """
You have access to the following tools:

1. print_state:
   Description: Shows the full conversation history.
   Usage: Say exactly: "USE_TOOL: PRINT_STATE"

2. RAG:
   Description: This tool searches and returns information from document.
   At first find the key words in the user's query related to stock market performance and use them in querry like bellow. 
   Usage: Say exactly: "USE_TOOL: RAG , KEY_WORDS: <key_words>"

Whenever the user requests something that matches a tool, call it by replying with:
USE_TOOL: <tool_name>
Do NOT explain or chat when using a tool.
"""

SYSTEM_PROMPT = msg_frame(
    "s",
    f"You are a helpful AI assistant named Alfred. Respond directly to the user.\n{TOOLS_DESCRIPTION}"
)


def redy_prompt(state: AgentState) -> str:
    messages = state["message"]
    prompt = ""
    for m in messages:
        prompt += f"{m['role']}: {m['content']}\n"
    prompt += "Assistant:"
    return prompt


async def chat(state: AgentState, writer: StreamWriter) -> AgentState:
    user_input = input("\n\nUser: ").strip()

    if user_input.lower() == "exit":
        state["message"].append(msg_frame("u", "exit"))
        return state

    state["message"].append(msg_frame("u", user_input))
    prompt = redy_prompt(state)

    print("AI: ", end="", flush=True)
    stream_text = ""
    async for chunk in llm.astream(prompt):
        if chunk.content:
            stream_text += chunk.content
            print(chunk.content, end="", flush=True)
    print()

    state["message"].append(msg_frame("a", stream_text))
    return state


@tool
def print_state(state: Annotated[dict, StateGraph]) -> str:
    """Shows the conversation history."""
    print("\nv--- Message ---v")
    for m in state["message"][1:]:
        print(f"{m['role']}: {m['content']}")
    print("^-------------^")
    return "State printed."

print_state_tool_node = ToolNode(tools=[print_state])

def should_continue(state: AgentState) -> str:
    messages = state["message"]
    if not messages:
        return "continue"

    last_message = messages[-1]["content"].strip().lower()

    if any(word in last_message for word in ["bye", "goodbye", "exit", "quit"]):
        return "end"
    elif last_message.startswith("use_tool:"):
        return "tools"
    else:
        return "continue"


async def tool_dispatcher(state):
    """
    Dynamically dispatches tools mentioned in the assistant's last message.

    Supports formats:
        USE_TOOL: <tool_name>
        USE_TOOL: <tool_name> , KEY_WORDS: <key_words>
    """
    pattern = re.compile(r"USE_TOOL:\s*(?P<tool>[A-Za-z_]\w*)(?:\s*,\s*KEY_WORDS:\s*(?P<keywords>.+))?", re.IGNORECASE)
    last_message = state["message"][-1]["content"].strip()
    match = pattern.search(last_message)

    if not match:
        return state

    tool_name = match.group("tool").strip().upper()
    keywords = match.group("keywords")
    if keywords:    keywords = keywords.strip()     
    else:           keywords = None

    print(f"\n🧰 Invoking tool: {tool_name} | Keywords: {keywords}")

    tool = TOOLS.get(tool_name)
    if not tool:
        state["message"].append(msg_frame("a", f"❌ Tool '{tool_name}' not found."))
        return state
    try:
        result = await tool.ainvoke({"state": state, "keywords": keywords})
    except Exception as e:
        state["message"].append(msg_frame("a", f"❌ Tool '{tool_name}' failed: {str(e)}"))
        return state


    state["message"].append(msg_frame("t", str(result)))

    print("\n🤖 LLM reasoning with tool output:\n")
    prompt = redy_prompt(state)
    stream_text = ""
    async for chunk in llm.astream(prompt):
        if chunk.content:
            stream_text += chunk.content
            print(chunk.content, end="", flush=True)


    state["message"].append(msg_frame("a", stream_text))
    return state

if #TODO  تنظیم ورودی و خروجی تول ها برای نود تول دیسپچ

graph = StateGraph(AgentState)
graph.add_node("chat", chat, stream=True)
graph.add_node("tools", tool_dispatcher)

graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat",
    should_continue,
    {
        "continue": "chat",
        "tools": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "chat")

compiled_graph = graph.compile()

def save_graph_visualization():
    with open("graph.png", "wb") as f:
        f.write(compiled_graph.get_graph().draw_mermaid_png())

async def main():
    print("Starting Alfred. Type 'exit' to quit.\n")
    state = {"message": [SYSTEM_PROMPT]}

    async for _ in compiled_graph.astream(state, stream_mode="custom"):
        pass  


#---------------------------------- RAG Example ----------------------------------#
collection_name = "stock_market"
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")

pdf_path = "Stock_Market_Performance_2024.pdf"

# === FUNCTION TO CREATE THE VECTOR DATABASE ===
def create_vectordb():
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Load and split the PDF
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load()
        print(f"✅ PDF loaded successfully with {len(pages)} pages")
    except Exception as e:
        raise RuntimeError(f"Error loading PDF: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    # Create and persist the database
    try:
        vectordb = Chroma.from_documents(
            documents=pages_split,
            embedding=embeder,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        print("✅ Chroma vector store created and persisted!")
    except Exception as e:
        raise RuntimeError(f"Error creating ChromaDB: {e}")

    return vectordb


# === MAIN LOGIC: CHECK OR CREATE DATABASE ===
def get_or_create_vectordb(collection_name :str):
    # Check if persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Load existing database if collection already exists
    try:
        existing_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeder,
            collection_name=collection_name,
        )
        # Check if collection actually has any embeddings
        if existing_db._collection.count() > 0:
            print(f"Found existing Chroma collection: '{collection_name}'")
            return existing_db
    except Exception:
        pass  # If any error, we'll just recreate it

    # Otherwise, create a new one
    print(f"No existing collection found. Creating new one: '{collection_name}'")
    return create_vectordb()


# === CONNECT RETRIEVER ===
vectordb = get_or_create_vectordb(collection_name)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

@tool
async def rag_tool(state: dict, keywords: str = None) -> str:
    """
    Searches for information from the Stock Market Performance 2024 document,
    returns retrieved context to the LLM for final answer generation.
    """
    # --- 1️⃣ Build the search query ---
    if keywords:
        query = keywords.strip()
    else:
        if len(state.get("message", [])) < 2:
            return "No recent user message found to perform retrieval."
        query = state["message"][-2]["content"].strip()

    print(f"\n🔍 Running RAG search for query: {query}")

    # --- 2️⃣ Retrieve documents (use MMR for better relevance) ---
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        return f"❌ Retrieval failed: {str(e)}"

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    # --- 3️⃣ Concatenate retrieved text as context ---
    context = "\n\n".join([doc.page_content for doc in docs])

    # --- 4️⃣ Add retrieved context to system messages for next LLM call ---
    state["message"].append(msg_frame("s", f"Context from Stock Market document:\n{context}"))

    # --- 5️⃣ Return instruction for next LLM response ---
    return (f"Now answer the user's question using ONLY the above context.\n user's question: '{query}'")


TOOLS = {
    "PRINT_STATE": print_state,
    "RAG": rag_tool,
}

if __name__ == "__main__":
    # save_graph_visualization()
    asyncio.run(main())