from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

#graph nodes
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]} 

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     #Run once create table
#     checkpointer.setup()

#     graph = builder.compile(checkpointer=checkpointer)

#     #Thread 1 (remembers)
#     t1 = {"configurable": {"thread_id": "thread-1"}}
#     graph.invoke({"messages": [{"role":"user", "content": "Hi my name is lubaba"}]}, t1)
#     out1 = graph.invoke({"messages": [{"role": "user", "content": "what is my name? "}]}, t1)
#     print("Thread-1:", out1["messages"][-1].content)



# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     #Run once create table
#     checkpointer.setup()
 
#     graph = builder.compile(checkpointer=checkpointer)

#     #Thread 1 (remembers)
#     t2 = {"configurable": {"thread_id": "thread-1"}}
#     graph.invoke({"messages": [{"role":"user", "content": "Hi my name is lubaba"}]}, t2)
#     out2 = graph.invoke({"messages": [{"role": "user", "content": "what is my name? "}]}, t2)
#     print("Thread-2:", out2["messages"][-1].content)




t1 = {"configurable": {"thread_id": "thread-1"}}
with PostgresSaver.from_conn_string(DB_URI) as cp:
        g = builder.compile(checkpointer=cp)
        snap = g.get_state(t1) #pulls from postgres
        msgs = snap.values.get("messages", [])
        print("Last message:", msgs[-1].content if msgs else "No messages found")


