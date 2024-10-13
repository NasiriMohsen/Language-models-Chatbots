import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

system_prompt = 'You are a friendly conversational chatbot with short responses'
history_length = 100

groq_chat = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name='llama3-8b-8192',
    temperature=0.9
)

history = ConversationBufferWindowMemory(k=history_length, memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=system_prompt
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),
    ]
)

conversation = LLMChain(
    llm=groq_chat,
    prompt=prompt,
    verbose=False,
    memory=history,
)

def stream_response(user_input,print_response = True):
    response_stream = groq_chat.stream(input = user_input)
    
    complete_response = ''
    if print_response:
        print("Chatbot: ", end="", flush=True) 
    for chunk in response_stream:
        content_value = chunk.content
        complete_response += content_value
        if print_response:
            print(content_value, end="", flush=True)  
    if print_response:
        print("")

    return complete_response

def full_response(user_input,print_response = True):
    response = conversation.predict(human_input = user_input)
    
    if print_response:
        print("Chatbot: ", response)
    
    return response

while True:
    user_question = input("User: ")
    stream_response(user_question)
