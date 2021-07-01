# -*- coding: utf-8 -*-
from flask import request
from flask_restx import Resource, Api, Namespace, fields
import torch
import numpy as np
import os
import pandas as pd
from analyze import make_result


Inference = Namespace(
    name="Inference",
    description="예측값을 얻기 위해 사용하는 API.",
)

inference_fields = Inference.model('Inference', {
    'data': fields.String(description='Input data file', required=True, example="Input data file for predict result")
})

inference_fields_with_data = Inference.inherit('Inference With ID', inference_fields, {
    'predict': fields.String(description='a Prediction'),
    'accuracy_score': fields.String(description='a accuracy_score'),
    'roc_auc_score': fields.String(description='a roc_auc_score'),
    'lgbm_plot_importance': fields.String(description='plot of lgbm_plot_importance'),
    'zero_one_distribution': fields.String(description='plot of zero_one_distribution'),
})

@Inference.route('')
class TodoPost(Resource):
    @Inference.expect(inference_fields)
    @Inference.response(201, 'Success', inference_fields_with_data)
    @Inference.response(500, 'Failed')
    def get(self):
        """Inference 반환."""
        df = pd.read_csv('./upload/test.csv')
        data_path = 'data_file'
        raw, score = make_result(data_path, df, True) # 0. plot_importance, 0 & 1 pred histogram
        
        print('결과', raw['pred'].iloc[-1])
        
        return {
            'prediction': raw['pred'].iloc[-1],
            'accuracy_score' : score['acc'],
            'roc_auc_score' : score['roc'],
            'lgbm_plot_importance' : "_00_lgbm_plot_importance.png",
            'zero_one_distribution': "_01_zero_one_distribution.png",
        }, 201, {'Access-Control-Allow-Origin': 'http://localhost:3000, https://baepo.netlify.app', 'Access-Control-Allow-Methods': 'GET'}
