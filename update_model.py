# importing the required module 
import pysftp 
import os

import time


local_dir='model_new'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
local_dir='model_new/exp/chain/tdnn/graph'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
local_dir='model_new/rescore'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
local_dir='model_new/rnnlm'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
    
# graph
# exp/chain/tdnn/graph

# rescore
# data/lang_test_rescore/G.fst

# rnnlm
# exp/rnnlm_out

def getting_(sftp,remote_path, local_path, file='__all__'):
    start_time = time.time()
    print('Preparing: '+local_path +'...')
    sftp.cwd('/root/vosk-model-en-us-0.22-compile/'+remote_path) 
    dir_struct = sftp.listdir_attr() 
    if file=='__all__':
      for attr in dir_struct: 
        print('dowloadng ' + attr.filename+ '  ...')
        sftp.get(attr.filename,local_path+attr.filename)
    else:
    	for attr in dir_struct: 
          if attr.filename==file:
             sftp.get(attr.filename,local_path+attr.filename)
             print('dowloadng ' + attr.filename+ '  ...')
          else:
          	print('File not found: '+ file)
    print("--OK-- %s seconds ---" % (time.time() - start_time))
 
my_Hostname = "193.168.46.90" 
my_Username = "root" 
my_Password = "Beget2022!" 
with pysftp.Connection( 
    host = my_Hostname, 
    username = my_Username, 
    password = my_Password 
    ) as sftp: 
    print("Connection succesfully established ... ") 
    getting_(sftp,'exp/chain/tdnn/graph/', 'model_new/exp/chain/tdnn/graph/')
    getting_(sftp,'data/lang_test_rescore/', 'model_new/rescore/', 'G.fst')
    getting_(sftp,'exp/rnnlm_out/', 'model_new/rnnlm/')
print('DONE')
         

