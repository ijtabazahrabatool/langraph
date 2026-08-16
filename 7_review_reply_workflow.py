import io
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal 
from pydantic import BaseModel, Field
from PIL import Image


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,
)

#create review Schema
class SentimentSchema(BaseModel):
    sentiment: Literal ["positive", "negative"] = Field(description="Sentiment of the review")

structured_sentiment_model = model.with_structured_output(SentimentSchema)

#create diagnosis Schema 
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description="The category of the issue mentioned in the review")
    tone: Literal["Angry", "Frustrated", "Confused", "Neutral", "Other"] = Field(description="The emotional tone expressed by the user ")
    urgency: Literal["High", "Medium", "Low"] = Field(description="How  urgent or critical the issue appears to be")

structured_diagnosis_model = model.with_structured_output(DiagnosisSchema)

#create state
class ReviewState(TypedDict):
    review: str
    sentiment: Literal ["positive", "negative"]
    diagnosis: dict 
    response: str
      
#create graph
graph = StateGraph(ReviewState)


#define nodes
#node 1: find sentiment of the review
def find_sentiment(state: ReviewState):
    review = state["review"]
    prompt = f"Find the sentiment of the following review: {review}."
    output = structured_sentiment_model.invoke(prompt)
    return {"sentiment": output.sentiment}

#node 2: positive response
def positive_response(state: ReviewState):
    review = state["review"]
    prompt = f"Write a warm thank-you message in response to the following review: {review}. Also kindly ask the user to leave feedback on our website."
    response = model.invoke(prompt).content
    return {"response": response}

#node 3: run_diagnosis
def run_diagnosis(state: ReviewState):
    review = state["review"]
    prompt = f"Diagnose the negative review based on the following criteria: issue type, tone, and urgency.The review is: {review}."
    output = structured_diagnosis_model.invoke(prompt)
    return {"diagnosis": output.model_dump()}

#node 4: negative response
def negative_response(state: ReviewState):
    review = state["review"]
    diagnosis = state["diagnosis"]
    prompt = f"you are support assistant. The user had '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'. Write an empathetic, helpful  resolution message"
    response = model.invoke(prompt).content
    return {"response": response}

#check sentiment  
def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    sentiment = state["sentiment"]
    if sentiment == "positive":
        return "positive_response"
    else: 
        return "run_diagnosis"


#add nodes
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

#add edge
graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)
graph.add_edge("positive_response", END)

#compile graph
workflow = graph.compile()

initial_state = {"review": "This product is amazing!"}
result = workflow.invoke(initial_state)
print(result)


#show graph
png_data = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data))
img.show()
