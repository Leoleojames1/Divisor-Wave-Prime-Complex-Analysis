from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)

# Read API key from file
with open("GPT3_API_key_ignored.txt", "r") as f:
    GPT3_API_key_ignored = f.read().strip()

# Set up headers with your API key
headers = {
    'Authorization': f'Bearer {GPT3_API_key_ignored}',
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

# Set up the home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    conversation = []
    if request.method == 'POST':
        # Get the user's input
        prompt = request.form['prompt']

        # Add the user's prompt to the conversation
        conversation.append(f'Human: {prompt}')

        # Set the prompt in the API data
        data['prompt'] = '\n'.join(conversation)

        # Send the API request
        response = requests.post('https://api.pawan.krd/v1/completions', headers=headers, data=json.dumps(data))

        # Get the text from the response
        text = response.json()['choices'][0]['text']

        # Add the AI's response to the conversation
        conversation.append(f'AI: {text}')

        # Render the result template with the response text and the conversation history
        return render_template('result.html', text=text, conversation=conversation)
    else:
        # Render the home template with an empty prompt and conversation history
        return render_template('home.html', prompt='', conversation=conversation)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)