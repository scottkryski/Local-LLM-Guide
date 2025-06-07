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
    model='llama3.2',
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