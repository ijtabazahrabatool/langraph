import io
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal 
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from PIL import Image



generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    project="imperial-medium-503420-s2",
    vertexai=True,
)

evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    project="imperial-medium-503420-s2",
    vertexai=True,
)

optimizer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    project="imperial-medium-503420-s2",
    vertexai=True,
)

#evaluation schema
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "need improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="Constructive feedback for the tweet.")
  #  score: int = Field(..., ge=0, le=5, description="Total score from rubric (0 to 5).")

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)




#create state
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["apporoved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iterations: int

#create graph
graph = StateGraph(TweetState)

#define nodes
#node 1: generate tweet 
def generate_tweet(state: TweetState): 
    
    #prompt for tweet generation
    messages=[
    SystemMessage(content="You are a funny and clever Twitter/X influencer."),
    HumanMessage(content=f"""
    write a short, original, and hilarious tweet on the topic: {state['topic']}.
    Rules:
    - Do NOT use question-answer format.
    - Max 280 characters.
    -Use Observational humor, irony, sarcasm, or cultural references.
    -Think in meme logic, punchlines, or relatable takes.
    -use simple day-to-day english
    This is version {state['iteration'] + 1}.
"""
    )
    ]
    #send generator_llm
    response = generator_llm.invoke(messages) 

    #return response
    return {"tweet": response.content}


#node 2: evaluate tweet 
def evaluate_tweet(state: TweetState):
    #prompt 
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweet based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:
Tweet: "{state['tweet']}

Use the criteria below to evaluate the tweet:

1.Originality - Is this fresh, or have you seen it hundered times before?
2.Humor - Did it genuinely make you smile, laugh, or chuckle?
3.Punchiness - It is short, sharp, and scroll stopping?
4.Virality Potential - Would people retweet or share it?
5. Format - Is it a well-formed tweet (not a setup punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format(e.g., "Why did..." or "What happens when... )
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Don't end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle
universe" or vague summaries)

### Respond ONLY in structured format:
- evalution: "approved" or "need improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
    ]
    response = structured_evaluator_llm.invoke(messages)
    return {"evaluation": response.evaluation, "feedback": response.feedback}

#node 3 optimize tweet
def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"
Topic: "{state['topic']}"
Original Tweet: "{state['tweet']}"

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = optimizer_llm.invoke(messages).content
    return {"tweet": response, "iteration": state["iteration"] + 1}

#check condition
def route_evaluation(state: TweetState) -> Literal["approved", "needs_improvement"]:
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "approved"   
    else: 
        return "needs_improvement"



#add node
graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

#add edges
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", route_evaluation, {"approved" : END, "needs_improvement": "optimize"})
graph.add_edge("optimize", "evaluate")


#compile graph
workflow = graph.compile()

initial_state = {"topic": "The role of AI in average people's lives" , "iteration": 1, "max_iterations": 5}
result = workflow.invoke(initial_state)
print(result)


#show graph
png_data = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data))
img.show()
