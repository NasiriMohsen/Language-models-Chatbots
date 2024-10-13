import os
from openai import OpenAI

# Set the API key
client = OpenAI(api_key=os.environ["GPT_API_KEY"])

def ask_gpt(question):
    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You do not use punctuation and your responses are short and to the point and you dont need to be formal."},
                {"role": "user", "content": question}
            ],
            n=1,
            max_tokens=100
        )
        
        # Extract and return the response
        answer = response.choices[0].message.content
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"

while True:
    question = input("You: ")
    response = ask_gpt(question)
    print(f"GPT's response: {response}")