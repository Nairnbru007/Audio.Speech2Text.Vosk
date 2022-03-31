import sys

try:
    file = sys.argv[1] # получаем 1й элемент списка (в примерах выше именно он отвечает за имя файла)
except IndexError: 
    sys.exit("error! choose the file!") 

from main import start_work
start_work(file,noise_reduce=False,cutting=False)
