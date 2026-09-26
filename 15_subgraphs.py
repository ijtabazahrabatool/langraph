from typing_extensions import TypedDict
from langgraph.graph import StateGraph,  START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class SubState(TypedDict):
    input_text: str
    translated_text: str



subgraph_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,)

def translate_text (state: SubState):
    prompt = f"""
    Translate the following text to Urdu.
    Keep it natural and clear. Do not add extra content 
    Text: {state['input_text']}
    """.strip()
    translated_text = subgraph_llm.invoke(prompt).content
    return {'translated_text': translated_text}
    
subgraph_builder = StateGraph(SubState)

subgraph_builder.add_node("translate_text", translate_text)

subgraph_builder.add_edge(START, "translate_text")
subgraph_builder.add_edge("translate_text", END)

subgraph = subgraph_builder.compile()


class ParentState(TypedDict):

    question: str
    answer_eng: str
    answer_urdu: str

parent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,
)


def generate_answer(state: ParentState):
    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly . \n\n Question: {state['question']}").content
    return {'answer_eng': answer}

def translate_answer(state: ParentState):

    #call the subgraph 
    result = subgraph.invoke({'input_text': state["answer_eng"]})
    return {'answer_urdu': result['translated_text']}

parent_builder = StateGraph(ParentState)

parent_builder.add_node("generate_answer", generate_answer)
parent_builder.add_node("translate_answer", translate_answer)

parent_builder.add_edge(START, "generate_answer")
parent_builder.add_edge("generate_answer", "translate_answer")
parent_builder.add_edge("translate_answer", END)

graph = parent_builder.compile()

print(graph.invoke({"question": "Tell me about stars in simple terms."}))
