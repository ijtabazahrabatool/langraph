from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

#create state
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int 

    sr: float
    bpb: float
    boundary_percentage: float
    summary: str

#create graph
graph = StateGraph(BatsmanState)

#create nodes 
def calculate_strike_rate(state: BatsmanState) -> BatsmanState:
    runs = state["runs"]
    balls = state["balls"]
    sr = runs/balls * 100
    return {'sr' : sr}

def calculate_boundary_percentage(state: BatsmanState) -> BatsmanState:
    runs = state["runs"]
    fours = state["fours"]
    sixes = state["sixes"]
    boundary_runs = (fours * 4) + (sixes * 6)
    boundary_percentage = (boundary_runs / runs) * 100
    return {'boundary_percentage' : boundary_percentage}

def calculate_balls_per_boundary(state: BatsmanState) -> BatsmanState:
   balls = state["balls"]
   fours = state["fours"]
   sixes = state["sixes"]

   boundaries = sixes + fours
   bpb = balls / boundaries
   return {'bpb' : bpb}

def summary(state: BatsmanState) -> BatsmanState:
    sr = state["sr"]
    bpb = state["bpb"]
    boundary_percentage = state["boundary_percentage"]
    summary = f"Summary of the batsman performance: \nStrike Rate: {sr}\nBalls per Boundary: {bpb}\nBoundary Percentage: {boundary_percentage}"
    return {'summary' : summary}


#add nodes
graph.add_node("calculate_strike_rate", calculate_strike_rate)
graph.add_node("calculate_boundary_percentage", calculate_boundary_percentage)
graph.add_node("calculate_balls_per_boundary", calculate_balls_per_boundary)
graph.add_node("summary", summary)

#add edges
graph.add_edge(START, "calculate_strike_rate")
graph.add_edge(START , "calculate_boundary_percentage")
graph.add_edge(START , "calculate_balls_per_boundary")
graph.add_edge("calculate_strike_rate", "summary")
graph.add_edge("calculate_boundary_percentage", "summary")
graph.add_edge("calculate_balls_per_boundary", "summary")
graph.add_edge("summary", END)

#compile graph
workflow = graph.compile()

#execute graph
initial_state = {'runs': 100, 'balls': 60, 'fours': 10, 'sixes': 5}
result = workflow.invoke(initial_state)

#print
print(f"Result: {result['summary']}")

#show graph
png_data = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data))
img.show()






