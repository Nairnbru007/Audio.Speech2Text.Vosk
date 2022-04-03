from flask import Flask, request, flash, render_template, make_response, Response
from flask_cors import CORS, cross_origin
from flask_restx import Api, Resource, fields, marshal, reqparse
import werkzeug
import sys
import logging
import time
import os
from werkzeug.utils import secure_filename
from main_server import start_work
#from utils import make_salted_hash, check_hashed_password, allowed_file, get_save_name, get_md5, save_uploadfile_to_backup

app = Flask('edges')

@app.before_first_request
def setup_logging():
    if not app.debug:
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

api = Api(app, version='1.0', title='Title API', description='API',)
CORS(app)
namespace = api.namespace('api', description='AllApis')
# ==========================================================================================
file_upload = reqparse.RequestParser()
file_upload.add_argument('file',
                          type=werkzeug.datastructures.FileStorage,
                          location='files',
                          required=False,
                          help='Image file cannot empty')


@api.route('/',methods = ['GET'])
class UP(Resource):
   def __init__(self):
      pass
   def get(self):
      headers = {'Content-Type': 'text/html'}
      return Response(render_template('index.html'),200,headers)

def allowed_files(file):
        if file.split('.')[-1].lower() in ['txt','avi','mp4','mp3','wav']:
                return True

@namespace.route('/mediafile')
class DetectApi(Resource):
   @namespace.expect(file_upload)
   def post(self):
        args = file_upload.parse_args()
        file = args['file']
        result = 'server get the file'
        if file and allowed_files(file.filename):
                filename = secure_filename(file.filename)
                path = time.strftime("%Y_%m_%d-%H_%M_%S")
                if not os.path.exists(path):
                   os.makedirs(path)
                file.save(os.path.join(path, filename))
                start_work(os.path.join(path, filename),noise_reduce=False,cutting=False)
                for i in os.listdir(path):
                   if i.endswith(".txt"):
                      out_txt = os.path.join(path, i)
                      with open(out_txt,'r') as fi:
                         result={'text':fi.read()}
                         print(result)
                         return {'results': result, 'error': ''}, 200 
                return {'results': result, 'error': ''}, 200 
        else:
                return {'results': '', 'error': 'file not allowed'}, 200 
        #return {'results': results, 'error': ''}, 200 
if __name__ == '__main__':
    app.run(port=5001, debug=False, host='0.0.0.0')  # , host='0.0.0.0'

						  
						  
					  
						  
