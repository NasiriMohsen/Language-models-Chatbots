import os
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

system_prompt = 'You are a friendly conversational chatbot'
history_length = 100

groq_chat = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name='llama3-8b-8192',
    temperature=0.9
)

history = ConversationBufferWindowMemory(k=history_length, memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

# Use the pipe operator | to chain the prompt and groq_chat
conversation = prompt | groq_chat

def full_response(user_input):
    history.chat_memory.add_message(HumanMessage(content=user_input))

    try:
        complete_response = ''
        print("Chatbot: ", end="", flush=True)
        
        response = conversation.stream({"human_input": user_input, "chat_history": history.chat_memory.messages})
        for chunk in response:
            content_value = chunk.content
            complete_response += content_value
            print(content_value, end="", flush=True)  

        print("")

    except KeyboardInterrupt:
        response.close()
        print("\n Interupted! ")
    
    history.chat_memory.add_message(AIMessage(content=complete_response))
    return complete_response

while True:
    user_question = input("User: ")
    response = full_response(user_question)