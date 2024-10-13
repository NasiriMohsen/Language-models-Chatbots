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


system_prompt = 'You are a friendly conversational chatbot'
history_length = 100


groq_chat = ChatGroq(
    groq_api_key = os.environ['GROQ_API_KEY'], 
    model_name = 'llama3-8b-8192',
    temperature=0.9
)

history = ConversationBufferWindowMemory(k = history_length, memory_key = "chat_history", return_messages=True)


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content = system_prompt
        ),
        MessagesPlaceholder(
            variable_name = "chat_history"
        ),
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ), 
    ]
)

conversation = LLMChain(
    llm = groq_chat,
    prompt = prompt,
    verbose = False,
    memory = history, 
)


while True:
    user_question = input("User: ")
    response = conversation.predict(human_input = user_question)
    
    print("Chatbot:", response)
