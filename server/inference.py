from flask import request
from flask_restx import Resource, Api, Namespace, fields
from args import parse_args
from dkt import trainer
import torch
import numpy as np
import os
from dkt.model import LSTM, BiLSTM, LSTMATTN, GRU, BiGRU, BERT, GRUATTN
import pandas as pd
from dkt.dataloader import Preprocess


Inference = Namespace(
    name="Inference",
    description="예측값을 얻기 위해 사용하는 API.",
)

inference_fields = Inference.model('Inference', {
    'data': fields.String(description='Input data', required=True, example="Input data for predict result")
})

inference_fields_with_data = Inference.inherit('Todo With ID', inference_fields, {
    'predict': fields.String(description='a Prediction')
})

@Inference.route('')
class TodoPost(Resource):
    @Inference.expect(inference_fields)
    @Inference.response(201, 'Success', inference_fields_with_data)
    @Inference.response(500, 'Failed')
    def post(self):
        """Inference 반환."""
        args = parse_args(mode='train')
        args.model = 'bert'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = device

        data = request.data.decode('utf-8').replace("\n", "\t").split("\t")
        column_names = ["userID","assessmentItemID","testId","answerCode","Timestamp","KnowledgeTag"]
        df = pd.DataFrame(columns = column_names)

        for idx in range(int(len(data)/len(column_names))):
            temp = {}
            for col in range(len(column_names)):
                temp[column_names[col]] = data[idx*len(column_names) + col]
            df = df.append(temp, ignore_index=True)

        preprocess = Preprocess(args)
        preprocess.load_test_data(df)
        test_data = preprocess.get_test_data()
        

        total_preds = trainer.inference(args, test_data)
        print('정답')
        print(total_preds)
        print('---------------')
        
        return {
            'prediction': total_preds,
        }, 201
