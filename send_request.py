import requests


payload = {"input": {"text_prompt": "The mind of the pig is full of curiosity and emotion - ukiyo-e style by Wassily Kandinsky"}}
response = requests.post("http://localhost:5000/predictions", json=payload)
breakpoint()