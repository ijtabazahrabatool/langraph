from typing_extensions import TypedDict
from langgraph.graph import StateGraph,  START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class ParentState(TypedDict):

    question: str
    answer_eng: str
    answer_urdu: str



parent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

subgraph_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,
)


def translate_text (state: ParentState ):
    prompt = f"""
    Translate the following text to Urdu.
    Keep it natural and clear. Do not add extra content 
    Text: {state['answer_eng']}
    """.strip()
    translated_text = subgraph_llm.invoke(prompt).content
    return {'translated_text': translated_text}
    
subgraph_builder = StateGraph(ParentState)

subgraph_builder.add_node("translate_text", translate_text)

subgraph_builder.add_edge(START, "translate_text")
subgraph_builder.add_edge("translate_text", END)

subgraph = subgraph_builder.compile()



def generate_answer(state: ParentState):
    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly . \n\n Question: {state['question']}").content
    return {'answer_eng': answer}



parent_builder = StateGraph(ParentState)

parent_builder.add_node("generate_answer", generate_answer)
parent_builder.add_node("translate", subgraph)

parent_builder.add_edge(START, "generate_answer")
parent_builder.add_edge("generate_answer", "translate")
parent_builder.add_edge("translate", END)

graph = parent_builder.compile()

print(graph.invoke({"question": "Tell me about stars in simple terms."}))
