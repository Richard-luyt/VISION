import requests
import json
import os

url = "http://127.0.0.1:8000/perdict_sequence"

image_paths = [f"data/0000/00000{i}.png" for i in range(10)] 

files = [('files', open(p, 'rb')) for p in image_paths]

boxes = [[10, 10, 100, 100] for _ in range(10)]
data = {'boxes_raw': json.dumps(boxes)}

print("sending request...")
response = requests.post(url, files=files, data=data)

print("the respond status:", response.status_code)
print("the response code:", response.json())