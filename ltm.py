from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
load_dotenv()
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END , MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


store = InMemoryStore()


SYSTEM_PROMPT_TEMPLATE = """You are helpful assistant with memory capabilities.
If User-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailtored
assistant that reflects the user's preferences, context, and past interactions.

If the User's name or relevant personal context is available, always personalize your responses by:
        Always Address the user by name(e.g., "Sure, Lubaba ") when appropriate
        Referencing known projects, tools, or preferences (e.g., "your MCP server python based project)
        Adjustion the tone to feel friendly, natural, and directly aimed at the user

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

memory_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if the memory is new, False if it is a duplicate of an existing memory")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")


memory_extractor = memory_llm.with_structured_output(MemoryDecision) 

MEMORY_POMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK: 
- Read the user's latest message.
- Extract user-specific into worth storing long-term (identity, stable preferences, ongoing projects/goals)
- For each extracted item, set is_new = true ONLY if it adds new information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new = false
- Keep each memory as a short atomic sentence
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return an empty list

"""



def remember_only_node(state: MessagesState, config: RunnableConfig, store: BaseStore):

    user_id = config["configurable"]["user_id"]

    
    namespace = ("user", user_id, "details")

    #take latest user message
    last_msg  = state["messages"][-1].content

    #LLM decides what to store
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "Extract LONG-TERM memories from the user's message.\n"
                    "Only store stable, user-specific info (identity, preferences, ongoing projects)" \
                    "Do NOT store transient info.\n"
                    "Return should_write = false if nothing is workth storing.\n"
                    "Each memory should be a short atomic sentence."
                )
            ),
            {"role": "user", "content": last_msg},
        ]
    )

    #write to store(LTM)
    if decision.should_write:
        for mem in decision.memories:
            store.put(namespace, str(uuid.uuid4()), {"data": mem})

    #IMPORTANT: we are NOT using memory, not even responding with the LLM.
    #We just return a fiexed acknowledgement.
    return {"messages": [{"role": "assistant", "content": "Noted."}]}


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
builder.add_node("remember_only", remember_only_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "remember_only")
builder.add_edge("remember_only", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store = store)



config = {"configurable": {"user_id": "u1"}}

result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "MY name is Alice and I like hiking."}]},
    config,

)

#print("\nAssistant:", result["messages"][-1].content)


result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "I like java programming"}]},
    config,

)

#print("\nAssistant:", result["messages"][-1].content)

result3 = graph.invoke(
    {"messages": [{"role": "user", "content": "I like java programming"}]},
    config,

)


for it in store.search(("user", "u1", "details")):
    print(it.value['data']) 