from groq import Groq
import os

class Robert:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def Ask_GROQ(self, question):
        self.chat_stream = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            model="llama3-8b-8192",
            temperature=0.9,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True
        )

        return self.chat_stream

#for chunk in chat_stream:
#    delta_content = chunk.choices[0].delta.content
#    if delta_content:
#        print(delta_content, end="")
#print()