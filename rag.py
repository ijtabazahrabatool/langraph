import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage   
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

os.environ['LANGCHAIN_PROJECT']='rag-tool'

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

#load document
loader = PyPDFLoader("intro_to_ml.pdf")
docs = loader.load()

#split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

#generate embeddings and store in FAISS vectorstore
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    project="imperial-medium-503420-s2",
    vertexai=True,
)
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k": 4})

@tool
def rag_tool(query):
    """
    Retrieve relevant information from the PDF document based on the query.
    use this tool when the user asks factual / conceptual questions
    that might be answered from the stored document.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }

tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

result = chatbot.invoke(
    {
        "messages":[
            HumanMessage(
                content="using pdf notes, explain how to find the ideal value of k in knn"
                )
        ]
    }
)


print (result["messages"][-1].content)