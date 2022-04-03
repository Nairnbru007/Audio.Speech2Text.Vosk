import sys
import requests
file_=sys.argv[1]
with open(file_, 'rb') as f:
    r = requests.post('http://127.0.0.1:5001/api/mediafile', files={'file': f})
print(r.text)


