from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END , MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

#Create LTM store + Seed memories (done BEFORE running the graph)
store = InMemoryStore() 

user_id = "u1"

#store user details as a single blob 
#can also split into multiple records ; this keep it easy
user_details = ("user", user_id, "details")
store.put(user_details, "profile 1", {"data": "Name: Lubaba"})
store.put(user_details, "profile 2", {"data": "profession : Software Engineer"})
store.put(user_details, "preference 1", {"data": "prefers detailed answers"})
store.put(user_details, "preference 2", {"data": "like example in simple language"})
store.put(user_details, "project 1", {"data": "building a python based MCP server"})


#----------------------------------------------------
#2) System prompt template (Your prompt)
#----------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are helpful assistant with memory capabilities.
If User-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailtored
assistant that reflects the user's preferences, context, and past interactions.

If the User's name or relevant personal context is available, always personalize your responses by:
        Always Address the user by name(e.g., "Sure, Lubaba ") when appropriate
        Referencing known projects, tools, or preferences (e.g., "your MCP server python based project)
        Adjustion the tone to feel friendly, natural, and directly aimed at the user
        Adjusting the tone to feel friendly, natural, and directly aimed at the user


Avoid generic phrasing when personalization is possible. for example, instead of "In TypeScript apps... 
say "Since your project is built with TypeScript...."

Use personalization especially in:
    Greetings and transitions
    Help or guidance tailored to tools and frameworks the user uses
    Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user's memory (which may be empty) is provided as: {user_details_content}
"""

#---------------------------------------------------------------------------
# 3)n Build graph: START -> chat -> END (read-only LTM)
#-------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)


def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):

    user_id = config["configurable"]["user_id"]

    #Read-only: fetch user details memory (no writes)
    user_details = ("user", user_id, "details")
    items = store.search(user_details)

    #convert memory items into a string blob for {user_details_content}
    #keep it dead simple for teaching.

    if items: 
        user_details_content = "\n".join(f"-{it.value.get('data', '')}" for it in items)
    else:
        user_details_content = "" #prompt says it may be empty

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_details_content=user_details_content
    )

    system_msg = SystemMessage(content=system_prompt)
    response =llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store = store)




#---------------------------------------
#4) Run it (provide user_id in config)
# -----------------------------------------
# 

config = {"configurable": {"user_id": "u1"}}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Explain gen ai in simple terms."}]},
    config,

)

print(result["messages"][-1].content)

