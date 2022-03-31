from flask import Flask, request, flash, render_template, make_response, Response
from flask_cors import CORS, cross_origin
from flask_restx import Api, Resource, fields, marshal, reqparse
import werkzeug
import sys
import logging

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
                          required=True,
                          help='Image file cannot empty')


@api.route('/',methods = ['GET'])
class UP(Resource):
   def __init__(self):
      pass
   def get(self):
      headers = {'Content-Type': 'text/html'}
      return Response(render_template('index.html'),200,headers)

@namespace.route('/123')
class DetectApi(Resource):
    @namespace.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        file_ = args['file']
        results = 'i send it'
        return {'results': results, 'error': ''}, 200
 
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers)
 
if __name__ == '__main__':
    app.run(port=5001, debug=False, host='0.0.0.0')  # , host='0.0.0.0'				  
						  
						  
					  
						  
