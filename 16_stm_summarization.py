from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

class ChatState(MessagesState):
      summary: str


#graph nodes
def chat_node(state: ChatState):
    messages = []

    if state["summary"]:
        messages.append({
            "role": "system",
            "content": f"Converstation summary: \n{state['summary']}",
        })
    messages.extend(state["messages"])
    print(messages)
    response = llm.invoke(messages)
    return {"messages": [response]} 

def summarize_converstation(state: ChatState):
    existing_summary = state["summary"]

     #Build summarization prompt
    if existing_summary:
        prompt = {
               f"Existing summary: \n:{existing_summary}\n\n"
               "Extend the summary using the new converstaion above"
          } 
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages_for_summary)

    #keep only two messsages verbatim
    messages_to_delete = state["messages"][-2:]
    return {
        "summary": response.content,
        "messages":[RemoveMessage(id = m.id) for m in messages_to_delete],
    }


def should_summarize(state: ChatState):
    return len(state["messages"]) >  6



builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_converstation)


builder.add_edge(START, "chat")
builder.add_conditional_edges(
    "chat",
    should_summarize,
    {
        True: "summarize",
        False: "__end__"
    }
)

builder.add_edge("summarize", "__end__")


checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "t1"}}

def show_state():
    snap = graph.get_state(config)
    vals = snap.values
    print("\n --- State ---")
    print("summary: ", vals.get("summary" , ""))
    print("num_messages: ", len(vals.get("messages", [])))
    print("messages:")
    for m in vals.get("messages" , []):
        print("-", type(m).__name__, ":" , m.content[:80])


    out = graph.invoke({"messages": [HumanMessage(content='Quantum Physics')] , 'summary': ''} , config = config)
    print(out)
    show_state()

    out = graph.invoke({"messages": [HumanMessage(content='Who is Alber Einstein?')] , 'summary': ''} , config = config)
    print(out)
    show_state()
    
