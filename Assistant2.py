import os
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

class Robert:
    def __init__(self, model='llama3-8b-8192', temp = 0.9, history_length = 10, system_prompt = "You are a friendly conversational chatbot"):
        
        self.history = ConversationBufferWindowMemory(k=history_length, memory_key="chat_history", return_messages=True)

        self.groq_chat = ChatGroq(
            groq_api_key = os.environ['GROQ_API_KEY'],
            model_name = model,
            temperature = temp
        )

        prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),])

        self.conversation = prompt | self.groq_chat

    def Stream_Response(self,user_input):
        self.history.chat_memory.add_message(HumanMessage(content=user_input))

        try:
            complete_content = ''
            print("Chatbot: ", end="", flush=True)
            response = self.conversation.stream({"human_input": user_input, "chat_history": self.history.chat_memory.messages})
            for chunk in response:
                content_value = chunk.content
                complete_content += content_value
                print(content_value, end="", flush=True)  
            print("")

        except KeyboardInterrupt:
            response.close()
            print("\n Interupted! ")

        self.history.chat_memory.add_message(AIMessage(content=complete_content))
        return complete_content
    
    def Final_Response(self,user_input):
        self.history.chat_memory.add_message(HumanMessage(content=user_input))
        response = self.conversation.invoke({"human_input": user_input, "chat_history": self.history.chat_memory.messages})
        self.history.chat_memory.add_message(AIMessage(content=response.content))
        return response.content


#while True:
#    user_question = input("User: ")

#    response = Stream_Response(user_question)
##      or
#    response = Final_Response(user_question)
#    print("Chatbot:", response)