from flask import request
from flask_restx import Resource, Api, Namespace, fields
from args import parse_args
from dkt import trainer
import torch
import numpy as np
import os
import pandas as pd
from dkt.dataloader import Preprocess
from analyze import make_result
from module_file.utils import feature_engineering_origin, feature_engineering, load_files
from module_file.inference import inference_plot
from module_file.draw_plot import test_group_draw, difference_by_count, zero_one_by_count, user_accuracy_prediction, total_time_with_prediction
from werkzeug.utils import secure_filename

Analysis = Namespace(
    name="Analysis",
    description="통계 plot을 제공해주는 API.",
)

analysis_fields = Analysis.model('Analysis', {
    'prob_count': fields.Integer(description='Count value', required=True, example="0"),
    'user_acc': fields.Integer(description='User_acc value', required=True, example="0.6"),
    'check': fields.String(description='check value', required=True, example="upper"),
})

analysis_fields_with_data = Analysis.inherit('Analysis images', analysis_fields, {
    'test_group_accrucy': fields.String(description='Image for test_group_accrucy'),
    'prediction_by_test_group': fields.String(description='Image for prediction_by_test_group'),
    'difference_by_count': fields.String(description='Image for difference_by_count'),
    'prediction_value_by_prob_count': fields.String(description='Image for prediction_value_by_prob_count'),
    'prediction_value_by_user_accuracy': fields.String(description='Image for prediction_value_by_user_accuracy'),
    'total_time_to_solve': fields.String(description='Image for total_time_to_solve'),
})

@Analysis.route('')
class AnalysisPost(Resource):
    @Analysis.expect(analysis_fields)
    @Analysis.response(201, 'Success', analysis_fields_with_data)
    @Analysis.response(500, 'Failed')
    def post(self):
        df = pd.read_csv('./upload/test.csv')
        data_path = 'data_file'
        raw, score = make_result(data_path, df, False) # 0. plot_importance, 0 & 1 pred histogram
        
        # 2. draw plot
        param = {'prob_count': int(request.form['prob_count']) if request.form['prob_count'] != '' else 0,
                 'user_acc': float(request.form['user_acc']) if request.form['user_acc'] != '' else 0.6,
                 'check': request.form['check'] if request.form['check'] != '' else 'upper'}  # or 'lower'

        # result analysis
        test_group_draw(raw)                                                              # 1. test_group
        difference_by_count(raw, count=param['prob_count'])                               # 2-1. 문제풀이 수 : 오차 체크
        zero_one_by_count(raw, count=param['prob_count'])                                 # 2-2. 문제풀이 수 : 특정 문제 수 이상
        # 모델 추론에 대한 원본 결과를 알고 싶을 때는 count=0
        user_accuracy_prediction(raw, user_acc=param['user_acc'], check=param['check'])   # 3. 유저의 정확도
        total_time_with_prediction(raw)
        
        
        return {
#             'lgbm_plot_importance' : "_00_lgbm_plot_importance.png",
#             'zero_one_distribution': "_01_zero_one_distribution.png",
            'test_group_accrucy' : "_10_test_group_accrucy",
            'prediction_by_test_group' : "_11_prediction_by_test_group.png",
            'difference_by_count': "_20_difference_by_count.png",
            'prediction_value_by_prob_count' : "_21_prediction_value_by_prob_count",
            'prediction_value_by_user_accuracy' : "_30_prediction_value_by_user_accuracy.png",
            'total_time_to_solve': "_40_total_time_to_solve.png",
        }, 201, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Method': 'POST'}
