import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings
from time import time
import time
import datetime
from datetime import datetime
import pickle
from glob import glob
from tqdm import tqdm
import joblib

filterwarnings('ignore')
pd.options.display.max_columns = 999


# 추후 시간이 오래걸리는 FE, row 변경에 영향도가 없는 FE 분리

def feature_engineering_origin(raw_test):
    test = raw_test.copy()
    test = test[test['userID'] == test['userID'].shift(-1)]
    test.sort_values(by=['userID', 'Timestamp'], inplace=True)
    test.join(test.groupby('userID')['answerCode'].count().to_frame('count'), on='userID', how='left')

    def convert_time(s):
        timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        return int(timestamp)

    test['time'] = test['Timestamp'].apply(convert_time)
    test['Timestamp'] = test['Timestamp'].astype('datetime64')
    test['test_group'] = test['testId'].astype('str').apply(lambda x: x[2])

    # add
    test['time'] = test['time'] - test['time'].shift(1)
    test.loc[0, 'time'] = 0

    test['testId'] = test['testId'].apply(lambda x: int(x[2:]))
    test['temp_test'] = test['testId'].shift(1)
    test.loc[0, 'temp_test'] = 0
    test['temp_test'] = test['temp_test'] - test['testId']
    test['temp_test'] = test['temp_test'].apply(lambda x: 0 if x != 0.0 else 1)
    test['time'] = test['time'] * test['temp_test']
    test['time_cum'] = test.groupby(['userID', 'testId'])['time'].transform(lambda x: x.cumsum())

    test['meantime_3'] = test.groupby('userID')['time'].rolling(3).mean().values  # rolling 앞의 3개 평균
    total_used_time = test.groupby('userID')['time'].cumsum().shift(1).to_frame('total_used_time').fillna(0)
    test = pd.concat([test, total_used_time], 1)

    return test


def feature_engineering(raw_test):
    test = raw_test.copy()

    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    test['user_correct_answer'] = test.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    test['user_total_answer'] = test.groupby('userID')['answerCode'].cumcount()
    test['user_acc'] = test['user_correct_answer'] / test['user_total_answer']

    total_prob = test.groupby('userID')['answerCode'].count().to_frame('total_prob').reset_index()
    test = pd.merge(test, total_prob, how='left', on='userID')
    test['total_prob'] = test['total_prob'] - test['user_correct_answer']

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    # 서비스 기준에 따라 변경됨
    correct_t = test.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = test.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    test = pd.merge(test, correct_t, on=['testId'], how="left")
    test = pd.merge(test, correct_k, on=['KnowledgeTag'], how="left")

    return test


def load_files(data_path):
    model = joblib.load(os.path.join(data_path, 'lgbm_model.pkl'))
    test_file = ''
#     test_file = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    with open(os.path.join(data_path, 'feature_pickle.pickle'), 'rb') as f:
        FEATURES = pickle.load(f)
    return model, test_file, FEATURES


