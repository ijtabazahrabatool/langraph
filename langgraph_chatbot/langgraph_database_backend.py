import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image
import sqlite3

load_dotenv()  # Load environment variables from .env file
os.environ["LANGCHAIN_PROJECT"] = 'chatbot-project' 
# define state
class ChatState(TypedDict):
    messages: Annotated [list[BaseMessage], add_messages]

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# checkpointer
checkpointer = SqliteSaver(conn=conn)

# create graph
graph = StateGraph(ChatState)

# define llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

# define node
def chat_node(state: ChatState):
    # take user query form state
    messages = state["messages"]

    # send to llm
    response = llm.invoke(messages)

    #response store in state 
    return {"messages": [response ]}

#add nodes 
graph.add_node("chat_node", chat_node)

#add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)

# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(print(checkpoint.config['configurable']['thread_id']))

#     return(list(all_threads))
