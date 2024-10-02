from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response based on user input
def chat_with_gpt_j(user_input):
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = chat_with_gpt_j(user_input)
    print("Robot: ", response)
