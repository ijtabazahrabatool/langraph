import io
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal 
from PIL import Image

#define state
class QuadState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    results: str 

#define graph
graph = StateGraph(QuadState) 

#define nodes
#node 1: show equation
def show_equation(state: QuadState):
    equation = f"The Quadratic equation is: {state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {"equation": equation}

#node 2: calculate discriminant
def calculate_discriminant(state: QuadState):
    discriminant = state['b'] ** 2 - 4 * state['a'] * state['c']
    return {"discriminant": discriminant}

#node 3: real roots
def real_roots(state: QuadState):
    discriminant = state['discriminant']
    root1 = (-state['b'] + discriminant ** 0.5) / 2 * state['a']
    root2 = (-state['b'] - discriminant ** 0.5) / 2 * state['a']
    results = f"The roots are real and distinct: {root1} and {root2}"
    return {"results": results}

#node 4 no real roots
def no_real_roots(state: QuadState):
    discriminant = state['discriminant']
    root1 = (-state['b'] + discriminant ** 0.5) / 2 * state['a']
    root2 = (-state['b'] - discriminant ** 0.5) / 2 * state['a']
    results = f"The roots are complex: {root1} and {root2}"
    return {"results": results}

#node 5 repeated roots
def repeated_roots(state: QuadState):
    discriminant = state['discriminant']
    root = -state['b'] / 2 * state['a']
    results = f"The roots are real and repeated: {root}"
    return {"results": results} 

#check condition 
def check_condition(state: QuadState) -> Literal["real_roots", "no_real_roots", "repeated_roots"]:
    discriminant = state['discriminant']
    if discriminant > 0:
        return "real_roots"
    elif discriminant == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"



#create nodes
graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("no_real_roots", no_real_roots)
graph.add_node("repeated_roots", repeated_roots )


#add edges
graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", check_condition)
graph.add_edge("real_roots", END)
graph.add_edge("no_real_roots", END)
graph.add_edge("repeated_roots", END) 

#compile graph
workflow = graph.compile()
initial_state = {'a': 4, 'b': 2, 'c': 4}
result = workflow.invoke(initial_state)
print(result)


#show graph
png_data = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data))
img.show()

