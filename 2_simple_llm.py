from langgraph.graph import StateGraph, START,END
#from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv


load_dotenv()


#model =model = ChatAnthropic( base_url="https://agentrouter.org",model="claude-opus-4-8")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


#create state
class LLMState(TypedDict):
    question: str
    answer: str



def llm_qa(state: LLMState) -> LLMState:
    #extract question from the state
    question = state["question"]
    #form a prompt
    prompt = f"Anwer the question in the humurous way: {question}"
    
    #ask question to the llm
    answer = model.invoke(prompt).content

    #update answer in the state
    state["answer"] = answer
    
    #return state
    return state


#create graph object
graph = StateGraph(LLMState)

#add node 
graph.add_node("llm_qa", llm_qa)

# add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

#compile graph
workflow = graph.compile()




#execute graph
initial_state = {'question': "why do programmers prefer dark mode?"}

final_state = workflow.invoke(initial_state)

print(f"Question: { final_state['question']}\n")
print(f"Answer: \n{ final_state['answer']}")


