def model_init(language='en'):
	#'ru'
	#шаг 0 Пипы и Воск
	#!pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
	#!pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# 	!pip install pytube
#  	!pip install youtube_dl
#  	!pip install vosk
#  	!pip install pydub
#  	!apt -qq install -y sox
#  	!pip install noisereduce
#  	!pip install spicy

    if not os.path.isdir('vosk-api'):#os.path.isfile('model'):
    	subprocess.call(['git','clone', 'https://github.com/alphacep/vosk-api'])
		#!git clone https://github.com/alphacep/vosk-api
	if not os.path.isdir('model'):#os.path.isfile('model'):
		if language=='ru':
			subprocess.call(['wget','https://alphacephei.com/kaldi/models/vosk-model-ru-0.22.zip'])
			#!wget https://alphacephei.com/kaldi/models/vosk-model-ru-0.22.zip
			subprocess.call(['unzip','vosk-model-ru-0.22.zip'])
			#!unzip vosk-model-ru-0.22.zip
			subprocess.call(['mv','vosk-model-ru-0.22', 'model'])
			#!mv vosk-model-ru-0.22 model
			os.remove('vosk-model-ru-0.22.zip')
		elif language=='en':
			subprocess.call(['wget','http://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip'])
			#!wget http://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
			subprocess.call(['unzip','vosk-model-en-us-0.22.zip']) 
			#!unzip vosk-model-en-us-0.22.zip
			subprocess.call(['mv','vosk-model-en-us-0.22', 'model'])
			#!mv vosk-model-en-us-0.22 model
			os.remove('vosk-model-ru-0.22.zip')