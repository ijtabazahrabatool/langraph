import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': "thread_1"}}

# Initialize chat history
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []
 
#loading the converstaion history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:

    #first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input })
    with st.chat_message('user'):
        st.text(user_input)
        
     
    # Simulate an assistant response
    
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk for message_chunk , metadata in chatbot.stream(
                  {"messages": [HumanMessage(content=user_input)]},
                    config  = {'configurable': {'thread_id': 'thread_ 1'}},
                    stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message })

 