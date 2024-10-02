import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

while True:
    user_input = input("You: ") 
    response = model.generate_content(user_input)
    print(f'Robot: {response.text}')