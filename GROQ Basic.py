from groq import Groq
import os

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def ask_groq(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system","content": "You do not use punctuation, your responses are short, you dont need to be formal and your name is robert.",},
            {"role": "user","content": question,}
        ],
        model="llama3-8b-8192",
        temperature=0.9,
    )
    return chat_completion.choices[0].message.content

while True:
    question = input("You: ")
    print(f"Response: {ask_groq(question)}")