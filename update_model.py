# importing the required module 
import pysftp 
import os

import time

local_dir_graph='/opt/vosk-model-en/model/graph/'
local_dir_rescore='/opt/vosk-model-en/model/rescore/'
local_dir_rnnlm='/opt/vosk-model-en/model/rnnlm/'
    
#local_dir_graph='model_new/exp/chain/tdnn/graph'
if not os.path.exists(local_dir_graph):
    os.makedirs(local_dir_graph)
#local_dir='model_new/rescore'
if not os.path.exists(local_dir_rescore):
    os.makedirs(local_dir_rescore) 
#local_dir='model_new/rnnlm'
if not os.path.exists(local_dir_rnnlm):
    os.makedirs(local_dir_rnnlm)
    
# graph path in beget
# exp/chain/tdnn/graph

# rescore path in beget
# data/lang_test_rescore/G.fst

# rnnlm path in beget
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
    getting_(sftp,'exp/chain/tdnn/graph/', local_dir_graph)
    getting_(sftp,'data/lang_test_rescore/', local_dir_rescore, 'G.fst')
    getting_(sftp,'exp/rnnlm_out/', local_dir_rnnlm)
print('DONE')
         

