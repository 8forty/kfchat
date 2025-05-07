import ollama
from ollama import ChatResponse

model_name = 'llama3.2:3b'
messages = [
    {'role': 'system', 'content': 'You are a helpful chatbot that talks in a professional manner.'},
    {'role': 'user', 'content': 'where is paris?'},
]
print(f'model: {model_name}')
client = ollama.Client(host='http://localhost:11434')

for i in range(2):
    print(f'--- loop {i}')
    chat_response: ChatResponse = client.chat(
        model=model_name,
        messages=messages,
        stream=False,
    )
    print(f'response: {chat_response.message.content}')
    client.generate(model=model_name, keep_alive=0.0)  # unload the model
