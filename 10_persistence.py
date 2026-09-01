import io
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver


# define llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

#create state
class JokeState(TypedDict):
    topic: str
    joke: str
    explaination: str

# check pointer
checkpointer = InMemorySaver()

#create graph
graph = StateGraph(JokeState)

#define nodes
#node 1: generate joke
def generate_joke(state: JokeState):
    topic = state["topic"]
    prompt = f"Generate a joke based on the topic: {topic}"
    joke = llm.invoke(prompt).content
    return {"joke": joke}

#node 2: explain joke
def explain_joke(state: JokeState):
    joke = state["joke"]
    prompt = f"Explain the following joke: {joke}"
    response = llm.invoke(prompt).content
    return {"explaination": response}

#add node
graph.add_node("generate_joke", generate_joke)
graph.add_node("explain_joke", explain_joke)

#add edges
graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "explain_joke")
graph.add_edge("explain_joke", END)

#compile graph
workflow = graph.compile(checkpointer=checkpointer)

#execute graph
initial_state = {"topic": "Isabella"}
result = workflow.invoke(initial_state)
print(result)