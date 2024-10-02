from transformers import pipeline

# Load GPT-Neo model from Hugging Face
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Function to generate a response based on user input
def chat_with_gpt_neo(user_input):
    response = generator(user_input, max_length=50, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = chat_with_gpt_neo(user_input)
    print("Robot: ", response)
