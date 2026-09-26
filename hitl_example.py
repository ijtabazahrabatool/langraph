import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import Any, add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel
import requests



load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = 'hitl-example' 


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question",
        "question": state["messages"][-1].content,
        "instructions": "Approve this question? yes/no"
    })

    if decision["approved"] == 'no':
        return {"messages": [AIMessage(content="Not Approved.")]}

    else: 
        response = llm.invoke(state["messages"])
        return {"messages": [response]}



#3. Build the graph: START -> chat -> END
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

#checkpointers
checkpointer = MemorySaver()

app = builder.compile(checkpointer=checkpointer)


#create a new thread id for this conversation
config = {'configurable': {'thread_id': '1234'}}

#------ Step 1: user ask a quesiton --------

initial_input = {
    "messages": [
        ("user", "Explain gradient descent in very simple terms.")
    ]
}


#Invoke the graph for the first time 
result = app.invoke(initial_input, config=config)

print(result)

message = result['__interrupt__'][0].value
print(message)


user_input = input(f"\nBackend message - {message} \n Approve this quesiton? (yes/no)")

#Resume the graph with the approval decision 
final_result = app.invoke(
    Command(resume={"approved": user_input}),
    config=config,
)

print(final_result)