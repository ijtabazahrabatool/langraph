import io
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image

# define state
class ChatState(TypedDict):
    messages: Annotated [list[BaseMessage], add_messages]

# check pointer
checkpointer = MemorySaver()

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

#loop for user input

 #def thread
thread_id = '1'
while True:
    user_message = input("Type here: ")
    print("User:" , user_message )
    if user_message.strip().lower() in ["exit", "quit", "bye"]:
        break

    #execute graph
    config = {'configurable': {'thread_id': thread_id}}
    result = chatbot.invoke({"messages": [HumanMessage(content=user_message)]}, config=config )
    print("AI: ", result["messages"][-1].content)
     

#show graph
png_data = chatbot .get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data)) 
img.show()  

