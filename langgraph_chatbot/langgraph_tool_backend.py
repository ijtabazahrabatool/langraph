import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

import sqlite3


load_dotenv() 

os.environ["LANGCHAIN_PROJECT"] = 'chatbot-project' 

# define llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)


#tools
search_tool = DuckDuckGoSearchRun(region="us-en")

#custom tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers. 
    Supported operations: add, subtract, multiply, divide. 
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, " result": result}
    except Exception as e:
        return {"error": str(e)} 


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=PRYK9UIQRAVD2S1G" 
    r = requests.get(url)
    return r.json() 

#Make tool list 
tools = [get_stock_price, search_tool, calculator]

#Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)



# define state
class ChatState(TypedDict):
    messages: Annotated [list[BaseMessage], add_messages]

#graph nodes
def chat_node(state: ChatState):
    """ LLM node that may answer or request a tool """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) #Executes tool calls

# checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# create graph
graph = StateGraph(ChatState)

#add nodes 
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node) 

#add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


# compile graph
chatbot = graph.compile(checkpointer=checkpointer)


