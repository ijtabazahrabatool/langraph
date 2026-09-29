from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import RemoveMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

#graph nodes
def chat(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]} 

def delete_old_messages(state: MessagesState):
    msgs = state["messages"]

    #if more then ten delete the first 6
    if len(msgs)> 10:
        to_remove = msgs[:6]
        return {"messages": [RemoveMessage(id = m.id) for m in to_remove]}

    return{}
      
            

builder = StateGraph(MessagesState)
builder.add_node("chat", chat) 
builder.add_node("cleanup", delete_old_messages)

builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup") #Run deletion after every node
builder.add_edge("cleanup","__end__")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1"}}


#Run Multiple turn
graph.invoke({"messages": [{"role": "user", "content": "I am Lubaba "}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Tell me about langgraph"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is langchain"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is quantum Mechanics? "}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is gen AI"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, config)



snap= graph.get_state(config)
print("Stored messages after cleanup" , len(snap.values["messages"]))