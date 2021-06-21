## https://justkode.kr/python/flask-restapi-2

from inference import Inference
from flask import Flask
from flask_restx import Resource, Api
# from todo import Todo
from todo import Todo
from inference import Inference
from analysis import Analysis
# from flask_cors import CORS, cross_origin


app = Flask(__name__, static_url_path='/static')
# CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(
    app,
    version='0.1',
    title="BoostCamp AI Tech Stage 4 DKT Baepo's API Server",
    description="Baepo's DKT API Server!",
    terms_url="/",
    contact="kdogyun@gmail.com",
    license="MIT"
)

# api.add_namespace(Todo, '/todos')
api.add_namespace(Inference, '/inference')
api.add_namespace(Analysis, '/analysis')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)