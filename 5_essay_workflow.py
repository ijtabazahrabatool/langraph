import io
import os
from tkinter import Image
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
from PIL import Image

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="imperial-medium-503420-s2",
    vertexai=True,
)




# create Evaluation schema
class Evaluation_schema(BaseModel):
     feedback: str = Field(description = "Detailed feedback for the essay")
     score: int = Field(description = "Score for the essay out of 10", ge=0, le=10 )

structured_model = model.with_structured_output(Evaluation_schema)

# create State
class EssayState(TypedDict):
     essay: str
     language_feedback: str
     clarity_feedback: str
     analysis_feedback: str
     final_feedback: str
     individual_scores: Annotated[list[int], operator.add]
     avg_score: float

#create graph
graph = StateGraph(EssayState)

#define nodes
#node 1: evaluate clarity of thought
def evaluate_clarity_of_thought(state: EssayState):
     essay = state["essay"]
     prompt = f"Evaluate the clarity of thought in the following essay: {essay}. Provide detailed feedback and a score out of 10."
     output = structured_model.invoke(prompt)
     return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

#node 2: evaluate depth of analysis
def evaluate_depth_of_analysis(state: EssayState):
     essay = state["essay"]
     prompt = f"Evaluate the depth of analysis in the following essay: {essay}. Provide detailed feedback and a score out of 10."
     output = structured_model.invoke(prompt)
     return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}
     
#node 3: evaluate language
def evaluate_language(state: EssayState):
     essay = state["essay"]
     prompt = f"Evaluate the language in the following essay: {essay}. Provide detailed feedback and a score out of 10."
     output = structured_model.invoke(prompt)
     return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

#node 4 final evaluation
def final_evaluation(state: EssayState):
     clarity_feedback = state["clarity_feedback"]
     analysis_feedback = state["analysis_feedback"]
     language_feedback = state["language_feedback"]
     
     #summarize feedback 
     prompt = f"Based on the following feedbacks, provide a summary feedback:\nClarity Feedback: {clarity_feedback}\nAnalysis Feedback: {analysis_feedback}\nLanguage Feedback: {language_feedback}\nIndividual Scores: {state['individual_scores']}."
     final_feedback = model.invoke(prompt).content

     #calculate average score
     avg_score = sum(state["individual_scores"])/ len(state["individual_scores"])
     return {'final_feedback': final_feedback, 'avg_score': avg_score}     

#add nodes
graph.add_node("evaluate_clarity_of_thought", evaluate_clarity_of_thought)
graph.add_node("evaluate_depth_of_analysis", evaluate_depth_of_analysis)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("final_evaluation", final_evaluation)

#add edges
graph.add_edge(START , "evaluate_clarity_of_thought")
graph.add_edge(START , "evaluate_depth_of_analysis")
graph.add_edge(START , "evaluate_language")
graph.add_edge("evaluate_clarity_of_thought", "final_evaluation")
graph.add_edge("evaluate_depth_of_analysis", "final_evaluation")
graph.add_edge("evaluate_language", "final_evaluation") 
graph.add_edge("final_evaluation",END)

#compile graph
workflow = graph.compile()

#execute graph
initial_state = {'essay': """A meteor shower is one of nature's most dazzling astronomical phenomena, taking place whenever planet Earth navigates through a dense stream of cosmic debris leftover along the orbital paths of comets and asteroids. As these celestial parent bodies travel near the inner Solar System, solar radiation heats their surfaces, causing volatile frozen gases to sublimate and release vast trails of dust, sand, and small rocky fragments known as meteoroids. These particle streams remain suspended in space along the comet's orbit for decades, centuries, or even millennia. Year after year, as Earth pursues its predictable path around the Sun, it periodically intersects these lingering debris clouds, setting the stage for a recurring meteor shower.

The stunning light show itself is a product of high-velocity physics occurring within Earth's upper atmosphere, primarily in the mesosphere between 75 and 100 kilometers above the surface. Despite most meteoroids being no larger than a tiny pebble or a grain of sand, they collide with our atmosphere at astonishing speeds ranging from roughly $11\text{ km/s}$ to $72\text{ km/s}$ (over 160,000 miles per hour). At these hypervelocities, intense ram pressure compresses and heats the air directly ahead of the incoming particle. This extreme heat rapidly vaporizes the meteoroid while simultaneously ionizing the surrounding atmospheric gases, generating a brilliant streak of glowing plasma that observers on the ground recognize as a shooting star or meteor. On rare occasions, larger particles produce extraordinary fireballs that briefly illuminate the entire landscape and leave glowing ionization trains lingering in the night sky.

A defining characteristic of any meteor shower is its radiant point, which gives rise to a fascinating optical illusion. When observing a meteor shower during its peak, all individual streaks of light appear to radiate outward from a single specific location in the night sky. In reality, the meteoroids are traveling through space along completely parallel paths relative to one another. Their apparent divergence from a central point is purely a perspective effect, identical to how parallel railroad tracks appear to converge toward a distant vanishing point on the horizon. Astronomy standardizes the naming of meteor showers based on the constellation in which this radiant point resides; for example, the famous Perseid meteor shower appears to originate from within the constellation Perseus, while the Geminids seem to shoot outward from Gemini.

Different meteor showers offer unique characteristics depending on the age, composition, and velocity of their parent bodies' dust trails. The Perseids, which peak every August, are spawned by Comet 109P/Swift-Tuttle and are celebrated for their high frequency and warm-weather viewing conditions. Conversely, the Geminids in December present an interesting anomaly because their parent body is not a traditional icy comet, but rather 3200 Phaethon, an unusual rock comet or near-Earth asteroid that sheds dense, slow-moving debris. Another notable display is the Leonid meteor shower in November, famous for historic "meteor storms" where hundreds of thousands of meteors per hour fell when Earth passed through exceptionally dense patches of fresh debris left by Comet 55P/Tempel-Tuttle.

Ultimately, meteor showers represent a captivating intersection between planetary science and public stargazing. Astronomers measure a shower's intensity using the Zenithal Hourly Rate (ZHR), an estimate of how many meteors a single observer could see per hour under ideal, clear, light-pollution-free skies with the radiant directly overhead. Studying the light spectra produced by vaporizing meteors allows scientists to analyze the chemical composition of primitive Solar System material without ever having to launch a sample-return space mission. For casual observers, understanding the mechanics behind these events transforms simple streaks of light into an appreciation for the dynamic, interconnected mechanics of our cosmic neighborhood.
"""}

result = workflow.invoke(initial_state)

#print
print(f"Feedback: {result['final_feedback']}, average score: {result['avg_score']}")


#show graph
png_data = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_data))
img.show()

