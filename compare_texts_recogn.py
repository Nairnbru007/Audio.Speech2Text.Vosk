import sys
import requests
from main_server import compare_texts
file_vosk=sys.argv[1]
file_orig=sys.argv[2]
compare_texts(file_vosk,file_orig)
