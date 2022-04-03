import sys
import requests
file_=sys.argv[1]
with open(file_, 'rb') as f:
    r = requests.post('http://127.0.0.1:5001/api/123', files={'file': f})
print(r.text)


