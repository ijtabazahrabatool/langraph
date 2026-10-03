from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

#----------------------------------------------------
#1) LTM store
#----------------------------------------------------

store = InMemoryStore()

#----------------------------------------------------
#2) LLMS
#.  -memory_llm: extract candidate memories + tell if each is new (no duplicate_of needed)
#----------------------------------------------------

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



def chat_creates_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):

    user_id = config["configurable"]["user_id"]

    namespace = ("user", user_id, "details")

    #)A load existing memories
    existing_memories = store.search(namespace)
    existing_texts = [it.value.get("data", "") for it in existing_memories if it.value.get("data")]
    user_details_content = "\n".join(f"-{it}" for it in existing_texts) if existing_texts else "(empty)"

    #B) Latest User Message
    last_text = state["messages"][-1]

    #C) LLM extracts memories + mark new vs duplicate 

    decision: MemoryDecision = memory_extractor.invoke(
    [
        SystemMessage(
            content=MEMORY_POMPT.format(
                user_details_content=user_details_content
            )
        ),
        {
            "role": "user",
            "content": f"USER MESSAGE: {last_text.content}",
        },
    ]
)

    #D) Store ONLY new memories
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text})

    return {"messages": [{"role": "assistant", "content": "Noted."}]}            



builder = StateGraph(MessagesState)
builder.add_node("chat", chat_creates_memory_node)
builder.add_edge(START, "chat")
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