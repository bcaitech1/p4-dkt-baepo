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
import time
import datetime
from datetime import datetime
import pickle
from glob import glob
from tqdm import tqdm
import joblib

filterwarnings('ignore')
pd.options.display.max_columns = 999

from module_file.utils import feature_engineering_origin, feature_engineering, load_files
from module_file.inference import inference_plot
from module_file.draw_plot import test_group_draw, difference_by_count, zero_one_by_count, user_accuracy_prediction, total_time_with_prediction



def make_result(data_path, df):
    plt.style.use('ggplot')
    model, raw_test, FEATURES = load_files(data_path) # 모델, test 원본 데이터, 사용할 FEATURES 불러오기

    raw_test = feature_engineering_origin(df) # FE1
    test_df = feature_engineering(raw_test)         # RE2

    raw, score = inference_plot(model, test_df, FEATURES) # raw : inference 후, plot_importance 상위 3개를 포함한 분석용 data
    return raw



# if __name__ == '__main__':

#     # 1. model inference
#     data_path = 'data_file'
#     raw = make_result(data_path)                                                         # 0. plot_importance, 0 & 1 pred histogram


#     # 2. draw plot
#     param = {'prob_count': 0,
#              'user_acc': 0.6,
#              'check': 'upper'}  # or 'lower'

#     # result analysis
#     test_group_draw(raw)                                                              # 1. test_group
#     difference_by_count(raw, count=param['prob_count'])                               # 2-1. 문제풀이 수 : 오차 체크
#     zero_one_by_count(raw, count=param['prob_count'])                                 # 2-2. 문제풀이 수 : 특정 문제 수 이상
#     # 모델 추론에 대한 원본 결과를 알고 싶을 때는 count=0
#     user_accuracy_prediction(raw, user_acc=param['user_acc'], check=param['check'])   # 3. 유저의 정확도
#     total_time_with_prediction(raw)
    
#     # 4. 문제풀이 총 시간

#     print(f"총 실행시간 : {int(time.time() - start)} s")
#     # raw 데이터로부터 feature_engineering_origin 이 오래걸림. pickle 로 중간부터 진행하면 절반 단축
