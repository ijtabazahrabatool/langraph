from langgraph.graph import StateGraph, START,END 
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
from PIL import Image
import io


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#create state
class BLOGState(TypedDict):
    topic: str
    outline: str
    content: str


#create graph
graph = StateGraph(BLOGState)

def generate_outline(state: BLOGState) -> BLOGState:
    topic = state["topic"]
    prompt = f"Generate a detailed outline for blog post on the topic:  {topic}"
    outline = model.invoke(prompt).content
    state["outline"] = outline
    return state

def generate_content(state: BLOGState) -> BLOGState:
    outline = state["outline"]
    topic = state["topic"]
    prompt = f"Generate a detailed content for blog post based on the outline:  {outline} along with the topic: {topic}"
    content = model.invoke(prompt).content
    state["content"] = content
    return state

#node 1: generate outline
graph.add_node("generate_outline", generate_outline)

#node 2: generate content
graph.add_node("generate_content", generate_content)

#add edges
graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "generate_content")
graph.add_edge("generate_content", END)

#compile graph
workflow = graph.compile()

#execute graph
initial_state = {'topic': "The impact of social media on mental health"}
result = workflow.invoke(initial_state)


print(f"Result: {result['content']}")

#show graph
png_data = workflow.get_graph().draw_mermaid_png()

img = Image.open(io.BytesIO(png_data))
img.show()



