import os
import yaml
import torch
from torch import package
from vosk import Model, KaldiRecognizer, SetLogLevel, SpkModel
import sys
import os
import wave
import json
import sys
import argparse
from statistics import mean
import json
#from statistics import quantiles #не работает
import numpy as np
import noisereduce as nr
import os
from scipy.io import wavfile
import time
import numpy
#from youtube_dl import YoutubeDL
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence,detect_silence,detect_leading_silence


def dowload_neuro_lang_model():
	torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
								   'latest_silero_models.yml',
								   progress=False)

	with open('latest_silero_models.yml', 'r') as yaml_file:
		models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
	model_conf = models.get('te_models').get('latest')


	model_url = model_conf.get('package')

	model_dir = "downloaded_model"
	os.makedirs(model_dir, exist_ok=True)
	model_path = os.path.join(model_dir, os.path.basename(model_url))

	if not os.path.isfile(model_path):
		torch.hub.download_url_to_file(model_url,
									   model_path,
									   progress=False)

	imp = package.PackageImporter(model_path)
	model = imp.load_pickle("te_model", "model")
	example_texts = model.examples
	return model

def apply_te(text, lan='ru'):
    #return model.enhance_text(text, lan)
    return dowload_neuro_lang_model.enhance_text(text, lan)


#from numba import jit, cuda
#code.interact(local=locals)
#Основная функция запуска
#@jit(target ="cuda")
def recognize(str_wav:str):
	#SetLogLevel(0)
	if not os.path.exists("model"):
	    print ("Please download the model (vosk-model-ru-0.10.zip) from https://github.com/alphacep/vosk-api/blob/master/doc/models.md and unpack as 'model' in the current folder.")
	    exit (1)
	wf = wave.open(str_wav, "rb")
	print('audio lenght ' + str(float(wf.getnframes()) / float(wf.getframerate())))
	if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
	#getnchannels() -> Возвращает количество аудиоканалов ( 1для моно, 2для стерео)
	#getsampwidth() -> Возвращает ширину образца в байтах
	#getcomptype() -> Возвращает тип сжатия ( 'NONE'это единственный поддерживаемый тип)
	#то есть все эти проверки на то, что нам на самом деле подсунули нужный формат
	    print ("Audio file must be WAV format mono PCM.")
	    print(wf.getnchannels())
	    print(wf.getsampwidth())
	    print(wf.getcomptype())
	    exit (1)
	
	#список для объединения результатов
	result = []#list()
	#Обученная модель для русского языка 
	model = Model("model")
	#wf.getframerate()->Возвращает частоту дискретизации.
	rec = KaldiRecognizer(model, wf.getframerate())
	rec.SetWords(True)
	#spk_model = SpkModel("model-spk")#
	#rec.SetSpkModel(spk_model)#
	while True:
		data = wf.readframes(4000)
		if len(data) == 0:
			break
		if rec.AcceptWaveform(data):
			jsonData = json.loads(rec.Result())
			result.append(jsonData)
	#print(jsonData)
	jsonData = json.loads(rec.FinalResult())
	result.append(jsonData)
	final = {"result":[],"text":""}
	count=0
	for i in result:
			#print(i)
			try: 
				i["result"]
			except: 
				result.pop(count)
				#print('---del---')
			else:
				temp1=i["result"]
				#final["result"].append(elem) #Это с автоматическим дележом воска на подмассивы по временным периодам.
				for elem in temp1:  ##
					final["result"].append(elem) ##
				temp2=i["text"]
				final["text"]=final["text"]+ " " + temp2 + '.' #точки от воска
			count += 1
	#text = jsonData['text']
	#final = {"result": result,"text":text}
	final = json.dumps(final,ensure_ascii=False)
	
	wf.close()
	return final







import asyncio
import websockets
import sys
import subprocess
#from pydub import AudioSegment


async def run_test(uri,file_name):
    list_output=[]	
    async with websockets.connect(uri) as websocket:
        file = file_name
        sound = AudioSegment.from_mp3(file)
        sound.export(file.split('.')[0]+".wav", format="wav")
        subprocess.call(['sox','-t','wav',file.split('.')[0]+".wav", file.split('.')[0]+"___ready___.wav", 'remix'>
        #file=file.split('.')[0]+"___ready___.wav"
        file=file.split('.')[0]+".wav"


        wf = open(file, "rb")
        #wrf = open(file.split('.')[0]+'__result.txt','w') 
        while True:
            data = wf.read(8000)

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
	asyncio.get_event_loop().run_until_complete(temp=run_test('ws://localhost:2700',str_wav))
	return temp

def recognize_without_all(str_wav:str):
	#SetLogLevel(0)
	if not os.path.exists("model"):
	    print ("Please download the model (vosk-model-ru-0.10.zip) from https://github.com/alphacep/vosk-api/blob/master/doc/models.md and unpack as 'model' in the current folder.")
	    exit (1)
	wf = wave.open(str_wav, "rb")
	if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
	#getnchannels() -> Возвращает количество аудиоканалов ( 1для моно, 2для стерео)
	#getsampwidth() -> Возвращает ширину образца в байтах
	#getcomptype() -> Возвращает тип сжатия ( 'NONE'это единственный поддерживаемый тип)
	#то есть все эти проверки на то, что нам на самом деле подсунули нужный формат
	    print ("Audio file must be WAV format mono PCM.")
	    print(wf.getnchannels())
	    print(wf.getsampwidth())
	    print(wf.getcomptype())
	    exit (1)
	
	#список для объединения результатов
	result = []#list()
	#Обученная модель для русского языка 
	model = Model("model")
	#wf.getframerate()->Возвращает частоту дискретизации.
	print(wf.getframerate())
	rec = KaldiRecognizer(model, wf.getframerate())
	rec.SetWords(True)	
	while True:
		data = wf.readframes(4000)
		if len(data) == 0:
			break
		if rec.AcceptWaveform(data):
			jsonData = json.loads(rec.Result())
			result.append(jsonData)
	jsonData = json.loads(rec.FinalResult())
	result.append(jsonData)
	final = {"result":[],"text":""}
	count=0
	for i in result:
			#print(i)
			try: 
				i["result"]
			except: 
				result.pop(count)
				#print('---del---')
			else:
				temp1=i["result"]
				final["result"].append(temp1) #Это с автоматическим дележом воска на подмассивы по временным периодам.
				#for elem in temp1:  ##
				#	final["result"].append(elem) ##
				temp2=i["text"]
				final["text"]=final["text"]+ " " + temp2 #точки от воска
			count += 1
	#text = jsonData['text']
	#final = {"result": result,"text":text}
	final = json.dumps(final,ensure_ascii=False)
	
	wf.close()
	return final

#распознать и сохранить в папке datajson json файл
def recogn_to_jsonfile(file_path,server=False):
  #file_path = "/content/data/ledenets.wav"
  if server==False:
  	fin = recognize(file_path)
  else:
	fin = recognize_server(file_path)
  #fin = recognize_without_all(file_path)
  with open(file_path.split('.')[0]+'.json', 'w') as outfile:
    json.dump(fin, outfile)

	#запятые
def com_neuro_vosk(fin):
  fin_text = ""
  temp_text = json.loads(fin)["text"]
  for predl in temp_text.split('.'):
    fin_text = fin_text + apply_te(predl) + " "
  return fin_text
#text_comm = apply_te(json.loads(fin)["text"])
def com_neuro_vosk_text(fin):
  fin_text = ""
  temp_text = fin
  for predl in temp_text.split('.'):
    fin_text = fin_text + apply_te(predl) + " "
  return fin_text

#!/usr/bin/env python

def txt2dict1(file):
    '''словарь из текстового файла;
    ключи - индексы предложений, построенных VOSK'''
    #data={}
    #with open(file) as json_file:
    #  print(json_file)
    #  data = json.load(json_file)
    data = json.loads(json.loads(open(file,"r").read()))
    #print(len(data))
    return data

def time_text1(src):
    '''cписок кортежей со словами и интервалами'''
    result = []
    #print(src['result'])
    tmp=[]
    for item in src['result']:
            tmp.append((item['word'], item['start'], item['end']))
    result.extend(tmp)
    return result

def time_list(stat_lst):
    '''список пауз между словами'''
    diff_lst = []
    for e, item in enumerate(stat_lst):
        if e < len(stat_lst) - 1:
            diff_lst.append(stat_lst[e+1][1] - stat_lst[e][2])
    return diff_lst[:]

def stat(pause_list):
    '''описательная статистика'''
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
    
    
    import matplotlib.pyplot as plt


    Y = list(range(1, len(X)+1))

    plt.scatter(Y,X)
    plt.show()
    
    
    return(text)


#иные функции
#удаление пунктуации и замена на точки окончание предложения
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

#преобразование mp3 в wav stereo в mono
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

#gain -1 rate 44100 dither -s
#Закачка из ютуба
# def ytb_d(url, file=""):
#   if file!="":
#       file=str(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))+'_'+url.split('/')[-1]+'.wav'
#   ydl_opts = {
#     'format': 'bestaudio/best',
#     'outtmpl': file,
#     'keepvideo': False,
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'wav',
#         'preferredquality': '192',
#       }],
#   }
#   try:
#         with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#             ydl.cache.remove()
#             info_dict = ydl.extract_info(url, download=False)
#             ydl.prepare_filename(info_dict)
#             ydl.download([url])
#             return True
#   except Exception:
#         return False 
#   #with YoutubeDL(ydl_opts) as ydl:
#   #  ydl.download([url])

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
#Основная функция запуска 
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
      #file___ready___.wav  to file_without_noise.wav to file___ready___.wav
      if noise_reduce==True and cutting==False:
          #print('noise_reduce=='+noise_reduce+ '|| cutting=='+cutting)
          rate, data = wavfile.read(file_split_point_0+"___ready___.wav")
          reduced_noise = nr.reduce_noise(y=data, sr=rate)
          wavfile.write(file.split('.')[0]+"_without_noise.wav", rate, np.asarray(reduced_noise, dtype=np.int16))
          os.remove(file.split('.')[0]+"___ready___.wav")
          os.rename(file.split('.')[0]+"_without_noise.wav", file_split_point_0+"___ready___.wav")
      print("--- %s seconds ---" % (time.time() - start_time))
      print('recognizing........')
      start_time = time.time()
      if cutting==True:
          print('cutting......')
          sound = AudioSegment.from_wav(file_split_point_0+"___ready___.wav")
          count_t=-16#-16
          count_s=500
          count_sp=50
          print(count_s)
          #chunks = split_on_silence(sound,min_silence_len=500,silence_thresh=count)
          detected=detect_silence(sound,min_silence_len=count_s,silence_thresh=count_t,seek_step=count_sp)
          #detected=detect_leading_silence(sound, silence_threshold=-200.0, chunk_size=10)
          print(detected)
          #
          #detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10)
          #detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1)
          while len(detected)<=2:
              if count_s>=300:
                  count_s=count_s-10
                  print("count_s= "+str(count_s))
                  print("count_t= "+str(count_t))
                  detected=detect_silence(sound,min_silence_len=count_s,silence_thresh=count_t,seek_step=count_sp)
                  print(detected)
              else:
                  count_t=count_t-1#1
                  count_s=500
          chunks = split_on_silence(sound,min_silence_len=count_s,silence_thresh=count_t)
          mass=[]
          print("--- %s seconds ---" % (time.time() - start_time))
          print(chunks)
          for i, chunk in enumerate(chunks):
              print(i)
              start_time = time.time()
              #----
              # min_silence_len must be silent for at least half a second    
              # silence_thresh consider it silent if quieter than -16 dBFS
              file_split_point_0_chunk_i=file_split_point_0+"__chunk{0}.wav".format(i)
              file_split_point_0_chunk_i_0=file_split_point_0_chunk_i.split('.')[0]
              chunk.export(file_split_point_0_chunk_i_0+"___ready___.wav", format="wav")
              #----
              if noise_reduce==True and cutting==True:
                  print('noise_reduce=='+noise_reduce+ '|| cutting=='+cutting)
                  rate, data = wavfile.read(file_split_point_0_chunk_i_0+"___ready___.wav")
                  reduced_noise = nr.reduce_noise(y=data, sr=rate)
                  wavfile.write(file_split_point_0_chunk_i_0+"_without_noise.wav", rate, np.asarray(reduced_noise, dtype=np.int16))
                  os.remove(file_split_point_0_chunk_i_0+"___ready___.wav")
                  os.rename(file_split_point_0_chunk_i_0+"_without_noise.wav", file_split_point_0_chunk_i_0+"___ready___.wav")
              recogn_to_jsonfile(file_split_point_0_chunk_i_0+"___ready___.wav")
              print("--- %s seconds ---" % (time.time() - start_time))

              curr_file_vosk = json.loads(json.loads(open(file_split_point_0_chunk_i_0+'___ready___.json',"r").read()))['text']
              vosk_main_avg = punct(main_avg(file_split_point_0_chunk_i_0+'___ready___.json'))
              vosk_main_qnt = punct(main_qnt(file_split_point_0_chunk_i_0+'___ready___.json'))
              #com_neuro_result = punct( apply_te(curr_file_vosk.replace('ё','е').replace('.',' '))  )
              #mass={'text_vosk' : curr_file_vosk.replace('ё','е'), 'text_vosk_neuro': com_neuro_vosk_text(curr_file_vosk.replace('ё','е')).lower(),'text_vosk_main_avg': vosk_main_avg.replace('ё','е').lower(), 'text_vosk_main_qnt':vosk_main_qnt.replace('ё','е').lower()}
              mass.append({'text_vosk' : curr_file_vosk.replace('ё','е'),'text_vosk_main_avg': vosk_main_avg.replace('ё','е').lower(), 'text_vosk_main_qnt':vosk_main_qnt.replace('ё','е').lower()})
              
              os.remove(file_split_point_0_chunk_i_0+"___ready___.wav")
              #os.remove(file_split_point_0_chunk_i_0+'___ready___.json')
          with open(file_split_point_0+'_texts_.json', 'w') as outfile:
              glob_mass=mass[0]
              for j in mass:
                 glob_mass[j]=''
                    
              for i in mass:
                    for j in i:
                        glob_mass[j]=glob_mass[j]+" "+i[j]
              json.dump(glob_mass, outfile)
          print("--- %s seconds ---" % (time.time() - start_time))
      else:    
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
    elif ".mp4" in file:  
      start_time = time.time()
      print('converting the file.....')
      subprocess.call(['ffmpeg','-i',file, '-ab', '160k', '-ac','2', '-ar', '44100', '-vn', file.split('.')[0]+'.wav'])
      file=file.split('.')[0]+".wav"
      print("--- %s seconds ---" % (time.time() - start_time))
 

def cut_wav(file):   
    sound = AudioSegment.from_wav(file)
    chunks = split_on_silence(sound,min_silence_len=500,silence_thresh=-16)
    # min_silence_len must be silent for at least half a second    
    # silence_thresh consider it silent if quieter than -16 dBFS
    for i, chunk in enumerate(chunks):
        chunk.export(file.split('.')[0]+"__chunk{0}.wav".format(i), format="wav")

#сравнения данных  
#'text_vosk', 'text_vosk_neuro' 'text_vosk_main_avg' 'text_vosk_main_qnt'
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
    #for k in i['text_orig'].split(' '):
    #  if "." in k:
    #    t_o.append(k.split('.')[0])
    #for k in i['text_vosk_main_qnt'].split(' '):
    #  if "." in k:
    #    t_v.append(k.split('.')[0])
        
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
    #for ii in diff:
    #    if (ii['good'] * 2.0/(ii['count_orig'] + ii['count_alg'])) >= 0:
    #        print(ii)
    #        print(ii['good'] * 2.0/(ii['count_orig'] + ii['count_alg']))
    #for ii in diff_words:
    #    if (len(ii['good']) * 2.0/(len(ii['count_orig']) + len(ii['count_alg']))) >= 0:
    #        print(ii)
    #print(len(diff))

    #ne_obn=0
    #lishnie=0
    #good=0
    #count_orig=0
    #for i in diff:
    #    if (i['good'] * 2.0/(i['count_orig'] + i['count_alg'])) >= 0:
    #        ne_obn=ne_obn+i['ne_obn']
    #        lishnie=lishnie+i['lishnie']
    #        good=good+i['good']
    #        count_orig=count_orig+i['count_orig']
 #[{'ne_obn': ne_obn, 'lishnie': lishnie, 'good': good, 'count_orig': count_orig}]
    #return([{'precision': good/(good+lishnie),'recal':good/(good+ne_obn)}])
    #for i in result:
        #print(i)
    return result

# def yotube_to_text(url):
#     file=str(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))+'_'+url.split('/')[-1]+'.wav'
#     if ytb_d(url,file)==False:
#         print('Error Downloads from Youtube')
#     else: start_work(file)

