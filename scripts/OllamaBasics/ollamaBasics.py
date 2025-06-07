import ollama

def print_lines():
    print("\n")
    for i in range(25):
        print("-", end='')
    print("\n")

# Chat example

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
)
print("\nChat response:")
print(response['message']['content'])

# Generate example
print_lines()
response = ollama.generate(
    model='llama3.2',
    prompt='Why is the sky blue?',
)
print("\nGenerate response:")
print(response['response'])

# Streaming example
print_lines()
stream = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)
print("\nStreaming response:")
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Options examples temp 0
print_lines()
response = ollama.generate(
    model='llama3.2',
    prompt='Why is the sky blue?',
    options={
        'temperature': 0,
        'num_ctx': 2048,
    }
)
print("\nOptions response, temperature=0:")
print(response['response'])

# Options examples temp 2
print_lines()
response = ollama.generate(
    model='llama3.2',
    prompt='Why is the sky blue?',
    options={
        'temperature': 2,
        'num_ctx': 2048,
    }
)
print("\nOptions response, temperature=2:")
print(response['response'])  # Example output will vary
