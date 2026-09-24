import os
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from pydantic import BaseModel
import requests
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = 'async-tools-project' 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)
 
client = MultiServerMCPClient(
    {
        "arith":{
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/taabi/Developer/langraph-tutorial/main.py"],
        },

        "expense":{
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)



# define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph(): 

    tools = await client.get_tools()

    #Make the LLM tool-aware
    llm_with_tools = llm.bind_tools(tools)

    #graph nodes
    async def chat_node(state: ChatState):
        """ LLM node that may answer or request a tool """
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node =   ToolNode(tools) #Executes tool calls

    #graph structure
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node) 

    #graph edges
    graph.add_edge(START, 'chat_node')
    #If the LLM asked for a tool, go to ToolNode, else finish
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()

    # #running the graph
    # out = await chatbot.ainvoke({"messages": [HumanMessage(content="Find modulous of 123254 and 23 and give answer like cricket commentor ")]})
    # print(out["messages"][-1].content)


    #running the graph
    out = await chatbot.ainvoke({"messages": [HumanMessage(content="Add an expense- Rs 500 udemy course on 23 sep 2026 ")]})
    print(out["messages"][-1].content)

if __name__ == '__main__':
    asyncio.run(main())
