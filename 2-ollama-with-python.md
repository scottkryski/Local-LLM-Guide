---
layout: default
title: "Part 2: Ollama with Python"
---

## Interacting with Ollama Programmatically

The `ollama` Python library provides a simple way to interact with Ollama programmatically.

### Setting up the Python Environment

First, create a virtual environment (good practice):

For Windows:

```bash
python -m venv venv
venv/Scripts/activate
```

For Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then simply install the `ollama` package:

```bash
pip install ollama
```

> ### **Note:**
>
> If you have a Nvidia GPU it's important to utilize it.
>
> Visit [Pytorch](https://pytorch.org/get-started/locally/) and following the instructions to get a build that works with your GPU's CUDA version.
>
> Otherwise, it will (normally) default to using your CPU which is slower. Ollama should already have support of Apple Silicon acceleration, but if you have a GPU its worth to use.

### Basic Interaction - Chat vs Generate

The library offers two main functions for interacting with models:

1. `ollama.chat()`: This is best for conversational interactions. It maintains a history of the conversation through a list of messages, which you pass with each request.
2. `ollama.generate()`: This is designed for single, non-conversational tasks like text completion, summarization, or simple questions. It's a more lightweight option when you don't need to maintain context.

### Chat Example:

```python
import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
)
print(response['message']['content'])
```

### Generate Example:

```python
import ollama

response = ollama.generate(
    model='llama3.2',
    prompt='Why is the sky blue?',
)
print(response['response'])
```

### Streaming Responses

For an interactive typing effect you can stream responses from the model. This works with both chat and generate:

```python
import ollama

stream = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

### Configuration Options:

You can customize the model's behavior by passing an `options` dictionary. This gives you fine-grained control over the generation process.

- `temperature` (e.g., `0.7`): Controls randomness. Higher values (like `1.0`) make the output more creative and random, while lower values (like `0.2`) make it more focused and deterministic.
- `num_ctx` (e.g., `4096`): Sets the context window size (in tokens). This determines how much of the previous conversation or prompt the model considers when generating a response.
- `top_k` (e.g., `40`): Reduces the pool of tokens to the `k` most likely candidates.
- `top_p` (e.g., `0.9`): Uses nucleus sampling, selecting from the smallest possible set of tokens whose cumulative probability exceeds `p`.

**Example with options:**

```python
import ollama

response = ollama.generate(
    model='llama3.2',
    prompt='Tell me a joke.',
    options={
        'temperature': 0.9,
        'num_ctx': 2048,
    }
)
print(response['response'])
```

### Structured Output with Pydantic

For tasks that require a predictable, structured output, you can force the model to respond with valid JSON that conforms to a specific schema.

This is useful for data extraction, classification, or interfacing with other tools.
The easiest way to do this is by defining your desired structure using a Pydantic class.

The ollama library will automatically convert the class into a JSON schema for the model.

1. Install pydantic with `pip install pydantic`
2. Define a pydantic class that inherits from `pydantic.BaseModel`
3. Pass the class to ollama.chat: The library handles the rest.

Additionally you can ask it to respond in JSON to help the model understand the request.

Setting the `temperature` to 0 also aids in getting a more deterministic output.

#### Example from Ollama's blog:

https://ollama.com/blog/structured-outputs

```python
from ollama import chat
from pydantic import BaseModel

class Pet(BaseModel):
  name: str
  animal: str
  age: int
  color: str | None
  favorite_toy: str | None

class PetList(BaseModel):
  pets: list[Pet]

response = chat(
  messages=[
    {
      'role': 'user',
      'content': '''
        I have two pets.
        A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
        I also have a 2 year old black cat named Loki who loves tennis balls.
      ''',
    }
  ],
  model='llama3.1',
  format=PetList.model_json_schema(),
)

pets = PetList.model_validate_json(response.message.content)
print(pets)
```

#### Output:

```json
pets=[
  Pet(name='Luna', animal='cat', age=5, color='grey', favorite_toy='yarn'),
  Pet(name='Loki', animal='cat', age=2, color='black', favorite_toy='tennis balls')
]
```

## Tool Calling

https://ollama.com/blog/tool-support

Some newer models have been trained to support tool calling, which allows them to request python functions to interact with external resources.

Tools can be added using the `tools` field when calling the LLM.

```python
import ollama
import json

# Example tool. You could integrate more features.
def get_weather(location: str):
    """
    Get the current weather for a specific location.
    In a real-world scenario, this would call a weather API.
    """
    if "boston" in location.lower():
        return json.dumps({"location": "Boston", "temperature": "10", "unit": "C"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "F"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content':
        'What is the weather in Boston?'}],

		# provide a weather checking tool to the model
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get the current weather for a specific location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and state, e.g., San Francisco, CA',
                        },
                    },
                    'required': ['location'],
                },
            },
        },
    ]
)

print(response['message']['tool_calls'])
```

### Conclusion

That's basically it for using Ollama's python library. Now we can move onto more advanced topics, such as Vector Stores and RAG.
