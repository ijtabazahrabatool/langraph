from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

MAX_TOKENS = 150
 

#graph nodes
def call_model(state: MessagesState):
    #Trim converstation  history -> last N messages that fits with in the token budget
    messages = trim_messages(
          state["messages"],
          strategy= "last",
          token_counter = count_tokens_approximately,
          max_tokens = MAX_TOKENS
          )
    print('current token count -> ', count_tokens_approximately(messages = messages))
    for message in messages:
          print(message.content)

    response = llm.invoke(messages)
    return {"messages": [response]} 

 
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")


checkpointer = InMemorySaver() # or
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "chat-1"}}
result = graph.invoke({"messages": [{"role": "user", "content": "MY name lubaba "}]}, config)
print(result["messages"][-1].content)



result = graph.invoke({"messages": [{"role": "user", "content": "I am learning python "}]}, config)
print(result["messages"][-1].content)

result = graph.invoke({"messages": [{"role": "user", "content": "can you explian short term memory"}]}, config)
print(result["messages"][-1].content)

result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print(result["messages"][-1].content)


for item in graph.get_state({"configurable": {"thread_id": "chat-1"}}).values["messages"]:
     print(item.content)
     print('-'*120)