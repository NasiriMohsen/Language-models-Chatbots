from transformers import pipeline

# Load the GPT-2 model for text generation
generator = pipeline('text-generation', model='gpt2')

# Function to generate a response based on user input
def chat_with_gpt(user_input):
    response = generator(
        user_input,
        truncation=True,         # Truncate the response to 50 tokens
        max_length=50,         # Limit the response length to 50 tokens
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256     # Pad with EOS token (end of sequence) to stop generation
    )
    return response[0]['generated_text']

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = chat_with_gpt(user_input)
    print("Robot: ", response)
