# importing the required module
import pysftp
import os
import time
import stat

my_Hostname = "193.168.46.90"
my_Username = "root"
my_Password = "Beget2022!"
# path in docker
local_dir_graph='/opt/vosk-model-en/model/graph/'
local_dir_rescore='/opt/vosk-model-en/model/rescore/'
local_dir_rnnlm='/opt/vosk-model-en/model/rnnlm/'
# path in beget
remote_dir_graph='exp/chain/tdnn/graph/'
remote_dir_rescore='data/lang_test_rescore/'
file1='G.fst'
file2='G.carpa'
remote_dir_rnnlm='exp/rnnlm_out/'

#local_dir_graph='model_new/exp/chain/tdnn/graph'
if not os.path.exists(local_dir_graph):
    os.makedirs(local_dir_graph)
#local_dir='model_new/rescore'
if not os.path.exists(local_dir_rescore):
    os.makedirs(local_dir_rescore)
#local_dir='model_new/rnnlm'
if not os.path.exists(local_dir_rnnlm):
    os.makedirs(local_dir_rnnlm)

def getting_(sftp,remote_path, local_path, file='__all__'):
    print('Preparing: '+local_path +'  ...')
    sftp.cwd('/root/vosk-model-en-us-0.22-compile/'+remote_path)
    dir_struct = sftp.listdir_attr()
    if file=='__all__':
      for attr in dir_struct:
        if stat.S_ISDIR(attr.st_mode):
          print('SUBDIR '+attr.filename)
          getting_(sftp,remote_path+attr.filename,local_path+attr.filename)
          print('END_SUBDIR '+attr.filename)
          sftp.cwd('/root/vosk-model-en-us-0.22-compile/'+remote_path)
        else:
          print('dowloadng ' + attr.filename+ '  ...')
          sftp.get(attr.filename,local_path+attr.filename)
    else:
        trigger=False
        for attr in dir_struct:
          if attr.filename==file:
             sftp.get(attr.filename,local_path+attr.filename)
             print('dowloadng ' + attr.filename+ '  ...')
             trigger=True
        if trigger==False:
          print('File not found: '+ file)


with pysftp.Connection(
    host = my_Hostname,
    username = my_Username,
    password = my_Password
    ) as sftp:
    print("Connection succesfully established ... ")
    print('.....1/4....')
    getting_(sftp,remote_dir_graph,local_dir_graph)
    print('.....2/4....')
    getting_(sftp,remote_dir_rescore, local_dir_rescore, file1)
    print('.....3/4....')
    getting_(sftp,remote_dir_rescore, local_dir_rescore, file2)
    print('.....4/4....')
    getting_(sftp,remote_dir_rnnlm, local_dir_rnnlm)
print('DONE')

