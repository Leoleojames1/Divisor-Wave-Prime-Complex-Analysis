"""
gpt_api_access.py
    gpt_api_access.py was designed using the ChatGPT API Free Projects by PawanOsman
    https://github.com/PawanOsman/ChatGPT#use-our-hosted-api-reverse-proxy
    This archive is a simple open-source proxy API that allows you to access OpenAI's ChatGPT API for free.
4/9/2023
@LeoBorcherding
"""

import requests
import json

# Set up headers with your API key
headers = {
    'Authorization': 'Bearer pk-cXkxkJMZvxteQWZSryGkIKyfNWJNwsMdgBfWQwYsVTbGOviZ',
    'Content-Type': 'application/json'
}

# Set up data with your API parameters
data = {
    'model': 'text-davinci-003',
    'temperature': 0.7,
    'max_tokens': 256,
    'stop': [
        'Human:',
        'AI:'
    ]
}

# Initialize memory variable
memory = ""

# Begin the conversation loop
while True:
    # Prompt the user for input
    print("#------------------------------------------------------------------------------")
    prompt = input("Human: ")

    # Exit the loop if the user is done with the conversation
    if prompt.lower() == "exit":
        print("Have a nice day!")
        break

    # Set the prompt in the API data, including the memory variable
    data['prompt'] = f'Human: {prompt}' #Here is our history: {memory}'

    # Send the API request
    response = requests.post('https://api.pawan.krd/v1/completions', headers=headers, data=json.dumps(data))

    # # Print the text from the response
    # text = response.json()['choices'][0]['text']

    # Get the text from the response and remove the "AI:" prefix and the new line character
    text = response.json()['choices'][0]['text'].strip("AI: \n")

    # # Update memory variable with previous response
    # memory = f'{memory} Human: {prompt} AI: {text}'

    print("#------------------------------------------------------------------------------")
    print(f"AI: {text}")

    if response.status_code != 200:
        print("Error: API request failed.")
        continue

    text_choices = response.json().get("choices", [])
    if not text_choices:
        print("Error: No response from API.")
        continue

    text = text_choices[0].get("text", "").strip("AI: \n")
    if not text:
        print("Error: Empty response from API.")
        continue

