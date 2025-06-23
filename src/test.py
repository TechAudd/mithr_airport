import requests

response = requests.get(
    'http://127.0.0.1:8000/session/',
    headers={'accept': 'application/json'}
)
if response.status_code == 200:
    data = response.json()

session_id = data.get('session_id')
print(f"Bot: {data['state']['next_question']}")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    params = {
        'session_id': session_id,
        'user_input': user_input
    }
    response = requests.post(
        'http://127.0.0.1:8000/chat/',
        params=params,
        headers={'accept': 'application/json'}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Bot: {data['state']['next_question']}")
    else:
        print("Error:", response.json().get('error', 'Unknown error occurred.'))

