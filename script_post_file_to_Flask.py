import sys
import requests
file=sys.argv[1]
with open(file, 'rb') as f:
    r = requests.post('http://127.0.0.1:6000/', files={'file': f})
print(r.text)
