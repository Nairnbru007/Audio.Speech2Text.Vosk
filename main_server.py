import os
import sys
import os
import wave
import json
#import argparse
from statistics import mean
import numpy as np
from scipy.io import wavfile
import time
import numpy
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence,detect_silence,detect_leading_silence


import asyncio
import websockets
#import sys
#import subprocess
#from pydub import AudioSegment


async def run_test(uri,file_name):
    list_output=[]      
    async with websockets.connect(uri) as websocket:
        file = file_name

        wf = wave.open(file, "rb")
        await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        buffer_size = int(wf.getframerate() * 0.2)
        while True:
            data = wf.readframes(4000)

            if len(data) == 0:
                break

            await websocket.send(data)
            #print (await websocket.recv(),file=wrf)
            list_output.append(await websocket.recv())

        await websocket.send('{"eof" : 1}')
        #print (await websocket.recv())
        list_output.append(await websocket.recv())
    return list_output

def recognize_server(str_wav:str):
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)
   result = asyncio.get_event_loop().run_until_complete(run_test('ws://localhost:2700',str_wav))
   print(result)
   final = {"result":[],"text":""}
   count=0
   for i in result:
                i=json.loads(i) #print(i)
                try: 
                        i["result"]
                        print(i["result"])
                except: 
                        result.pop(count)
                        print('---del---')
                else:
                        temp1=i["result"]
                        for elem in temp1:
                                print(elem)  ##
                                final["result"].append(elem) ##
                        temp2=i["text"]
                        final["text"]=final["text"]+ " " + temp2 + '.'
                count += 1
                print(count)
                #print(i)
   final = json.dumps(final,ensure_ascii=False)

   return final

def recogn_to_jsonfile(file_path,server=True):
  #file_path = "/content/data/ledenets.wav"
  if server==False:
   fin = recognize(file_path)
  else:
   fin = recognize_server(file_path)
  #fin = recognize_without_all(file_path)
  with open(file_path.split('.')[0]+'.json', 'w') as outfile:
    json.dump(fin, outfile)
  
    
    
def txt2dict1(file):
    #data={}
    #with open(file) as json_file:
    #  print(json_file)
    #  data = json.load(json_file)
    data = json.loads(json.loads(open(file,"r").read()))
    #print(len(data))
    return data

def time_text1(src):
    result = []
    #print(src['result'])
    tmp=[]
    for item in src['result']:
            tmp.append((item['word'], item['start'], item['end']))
    result.extend(tmp)
    return result

def time_list(stat_lst):
    diff_lst = []
    for e, item in enumerate(stat_lst):
        if e < len(stat_lst) - 1:
            diff_lst.append(stat_lst[e+1][1] - stat_lst[e][2])
    return diff_lst[:]

def stat(pause_list):
    return {'mean': mean(pause_list), 'median': median(pause_list),
            'mode': mode(pause_list), 'quantiles': quantiles(pause_list), 
            'stdev': stdev(pause_list)
           }

def main_avg(file):
    src = txt2dict1(file)
    stat_lst = time_text1(src)
    words = time_list(stat_lst)
    src_data = [w for w in words if w]
    mn = mean(src_data)
    words.append(0.0)
    text = ''
    for word, time in zip(stat_lst, words):
        text += word[0]
        if time > mn:
            text += '. '
        elif time > mn * 0.5:
            text += ', '
        else:
            text += ' '
    text = text[:-1] + '.'
    return(text)

def main_qnt(file):
    X=[]
    src = txt2dict1(file)
    stat_lst = time_text1(src)
    words = time_list(stat_lst)
    src_data = [w for w in words if w]
    qs = np.quantile(src_data,(0.25, 0.50, 0.75))
    print(qs)
    #qs = quantile(src_data)
    words.append(0.0)
    text = ''
    for word, time in zip(stat_lst, words):
        #
        X.append(time)
        #
        text += word[0]
        if time > qs[2]:
            text += '. '
        elif time > qs[1]:
            text += ', '
        else:
            text += ' '
    text = text[:-1] + '.'
    
    #import matplotlib.pyplot as plt
    #Y = list(range(1, len(X)+1))
    #plt.scatter(Y,X)
    #plt.show()
    
    return(text)
  
def punct(text):
  #punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
	punctuations = '''!()---[]{};:"\,<>./?@#$%^&*_~'''
	punctuation_replace = '!:;.?'
	alphabet = "'"+'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	inp_str = text
	no_punc = ""
	last=""
	for char in inp_str:
		if char not in punctuations:
			if char in alphabet:
				if last=='.':
					no_punc = no_punc + " " + char
				else:
					no_punc = no_punc + char
				last=""	
			elif char==" ":
				if char!=last:
					no_punc = no_punc + char
					last=" "
				else:
					pass
			else:
				last=""
				#print('new symbol '+char)
		else:
			if char in punctuation_replace:
				if last!='.':
					char = '.'
					no_punc = no_punc + char
					last = "."
				else:
					pass
			else:
				pass
	return no_punc


def mp3_wav(file):
  #sound = AudioSegment.from_waw("123.wav")
  sound = AudioSegment.from_mp3(file)
  #sound = sound.set_channels(1)
  #sound.export("../content/data/"+file.split('.')[0]+".wav", format="wav")
  #sound = sound + 3
  sound.export(file.split('.')[0]+".wav", format="wav")

def mono_wav(file):
  #subprocess.call(['sox', file, file.split('.')[0]+"_mono.wav", 'remix', '1,2'])
  #subprocess.call(['sox', '-q','--multi-threaded',file, file.split('.')[0]+"___ready___.wav", 'remix', '1,2'])
  subprocess.call(['sox','-t','wav',file, file.split('.')[0]+"___ready___.wav", 'remix', '1,2','gain', '-1', 'rate', '44100', 'dither', '-s'])
  
def to_up(text):
    trigger=True
    text_=''
    if text[0]==' ':
        start=1
    else:
        start=0
    for i in range(start,len(text)-1):
        temp=text[i]
        if text[i]=='.':
            trigger=True
        if text[i] in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz' and trigger==True:
            temp=text[i].upper()
            trigger=False
        text_=text_+temp
    return text_

import shutil

def start_work(file,noise_reduce=False,cutting=False):
    print('noise reduce='+str(noise_reduce)+', cutting='+str(cutting))
    glob_time = time.time()
    
    os.rename(file,file.replace(' ','_'))
    file=file.replace(' ','_')
    
    if ".wav" in file:
      file_split_point_0=file.split('.')[0]
      #make copy file to file___orig___.wav
      shutil.copy2(file, file_split_point_0+"___orig___.wav")
      start_time = time.time()
      print('preparing the file.....')
      #file to mono file___ready___.wav
      mono_wav(file_split_point_0+".wav")
      if os.path.isfile(file_split_point_0+"___ready___.wav")==False:
            shutil.copy2(file, file_split_point_0+"___ready___.wav")
      print("--- %s seconds ---" % (time.time() - start_time))
      print('recognizing........')
      start_time = time.time()
      
      
      recogn_to_jsonfile(file.split('.')[0]+"___ready___.wav",server=True)
      print("--- %s seconds ---" % (time.time() - start_time))
      print('punctuation........')
      start_time = time.time()
      curr_file_vosk = json.loads(json.loads(open(file.split(".")[0]+'___ready___.json',"r").read()))['text']
      vosk_main_avg = punct(main_avg(file.split(".")[0]+'___ready___.json'))
      vosk_main_qnt = punct(main_qnt(file.split(".")[0]+'___ready___.json'))
      #com_neuro_result = punct( apply_te(curr_file_vosk.replace('ё','е').replace('.',' '))  )
      #mass={'text_vosk' : curr_file_vosk.replace('ё','е'), 'text_vosk_neuro': com_neuro_vosk_text(curr_file_vosk.replace('ё','е')).lower(),'text_vosk_main_avg': vosk_main_avg.replace('ё','е').lower(), 'text_vosk_main_qnt':vosk_main_qnt.replace('ё','е').lower()}
      mass={'text_vosk' : curr_file_vosk.replace('ё','е'),'text_vosk_main_avg': vosk_main_avg.replace('ё','е').lower(), 'text_vosk_main_qnt':vosk_main_qnt.replace('ё','е').lower()}
      with open(file.split(".")[0]+'_texts_.json', 'w') as outfile:
        json.dump(mass, outfile)
      print("--- %s seconds ---" % (time.time() - start_time))
      os.rename(file.split('.')[0]+"___ready___.json",file.split('.')[0]+"_vosk_.json")
      
      
      
      os.remove(file.split('.')[0]+".wav")
      os.rename(file.split('.')[0]+"___ready___.wav",file.split('.')[0]+".wav")
      print(file+" FINISH --- %s seconds" % (time.time() - glob_time))
      
      file_j =file.split('.')[0]+"_vosk_.json"
      file_text =file.split('.')[0]+".txt"
      text_file = open(file_text, "w")
      text_file.write( to_up( punct(main_qnt(file_j)) ) )
      text_file.close()
    
    elif ".mp3" in file:
      start_time = time.time()
      print('converting the file.....')
      mp3_wav(file)
      file=file.split('.')[0]+".wav"
      print("--- %s seconds ---" % (time.time() - start_time))
      start_work(file,noise_reduce=noise_reduce,cutting=cutting)
    elif ".avi" in file:  
      start_time = time.time()
      print('converting the file.....')
      subprocess.call(['ffmpeg','-i',file, '-ab', '160k', '-ac','2', '-ar', '44100', '-vn', file.split('.')[0]+'.wav'])
      file=file.split('.')[0]+".wav"
      print("--- %s seconds ---" % (time.time() - start_time))
      start_work(file,noise_reduce=noise_reduce,cutting=cutting)
    elif ".mp4" in file:  
      start_time = time.time()
      print('converting the file.....')
      subprocess.call(['ffmpeg','-i',file, '-ab', '160k', '-ac','2', '-ar', '44100', '-vn', file.split('.')[0]+'.wav'])
      file=file.split('.')[0]+".wav"
      print("--- %s seconds ---" % (time.time() - start_time))
      start_work(file,noise_reduce=noise_reduce,cutting=cutting)
      
      
def compare_result(json_file,text_file):
    txt = punct(punct(open(text_file,'r').read()).lower())
    print(txt)
    mass = json.loads(open(json_file,"r").read())
    diff=[]
    diff_words=[]
    result=[]
    t_o=[]
    for k in txt.split('.'):
        t_o.append(k.replace(' ','')[0:3])
    for i in mass:
        t_v=[]
        print(mass[i])    
        for k in punct(mass[i]).split('.'):
            t_v.append(k.replace(' ','')[0:3])
        ne_obn = list(set(t_o[:]) - set(t_v[:]))
        lishnie = list(set(t_v[:]) - set(t_o[:]))
        good = list(set(t_o[:]) & set(t_v[:]))
        count_orig = list(t_o[:])
        count_alg = list(t_v[:]) 

        diff.append({'name':i,'ne_obn':len(ne_obn), 'lishnie':len(lishnie), 'good':len(good), 'count_orig': len(count_orig), 'count_alg': len(count_alg)})
        diff_words.append({'name':i,'ne_obn':ne_obn, 'lishnie':lishnie, 'good':good, 'count_orig':count_orig, 'count_alg': count_alg})
        result.append({'name':i,'precision': len(good)/(len(good)+len(lishnie)),'recal':len(good)/(len(good)+len(ne_obn))})#,'text':mass[i]})
    return result
    
