import streamlit as st
from langgraph_backend import chatbot , llm
from langchain_core.messages import HumanMessage
import uuid 

#****************************** Utitlity functions ********************************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id() 
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = [] 

def generate_thread_title(user_msg) -> str:
    prompt = f"Summarize this user query into a concise 3-5 word title. Do not use quotes, punctuation, or preamble:\n\n{user_msg}"
    response = llm.invoke(prompt)
    return response.content.strip().replace("\n", " ")

def add_thread(thread_id , first_message: str):
    #Adds a new thread to the session state if it doesn't already exist.
    thread_exists = any(t['id'] ==  thread_id for t in st.session_state['chat_threads'])

    if not thread_exists:
        st.session_state['chat_threads'].append(
            {
                'id': thread_id,
                'title': generate_thread_title(first_message)
            }
        )
        return True
    return False
    

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']





#**************************************** Session Setup ****************************************
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state: 
    st.session_state['chat_threads'] = []





#**************************************** Sidebar UI *************************************************
  
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('+ New Chat'):
    reset_chat()

st.sidebar.header("My Conversations")




for threads in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button (threads['title'], key=f"btn_{threads['id']}"):
        st.session_state['thread_id'] = threads['id']
        messages = load_conversation(threads['id'])

        temp_message  = []

        for msg in messages: 
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_message.append({'role': role , "content": msg.content })

        st.session_state['message_history'] = temp_message
      

       
# **************************************** Main UI ****************************************************
 
#loading the converstaion history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])



if user_input := st.chat_input("Type here"):

    #check if the thread is new and add it to the session state
    is_new = add_thread(st.session_state['thread_id'], user_input)
    
    #first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input })
    with st.chat_message('user'):
        st.text(user_input)
        

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Simulate an assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk for message_chunk , metadata in chatbot.stream(
                  {"messages": [HumanMessage(content=user_input)]},
                    config  = CONFIG,
                    stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message })

    if is_new:
        st.rerun()  # Refresh the page to update the sidebar with the new thread
