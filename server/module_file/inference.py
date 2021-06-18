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


def inference_plot(model, test_df, FEATURES):

    data = {}

    def set_df(test, features):
        test2 = test[test['userID'] != test['userID'].shift(-1)]
        y = test2['answerCode']
        return test2, test2[features], y
    
    def zero_one_distribution(df):
        zero = df[df['answerCode']==0]
        one = df[df['answerCode']==1]
        plt.figure(figsize=(16,5))
        plt.hist(zero['pred'], bins=20,rwidth=0.9, alpha=0.5, label='predict 0', color='blue', edgecolor='black')
        plt.hist(one['pred'], bins=20,rwidth=0.9, alpha=0.4, label='predict 1', color='red', edgecolor='black')
        plt.legend(loc='upper left')
        plt.title('Prediction for 0 and 1 - Distribution', loc='left', size=16)
        plt.suptitle("Model Inference", fontweight='bold', size=20)
        plt.grid()
        plt.savefig('static/_01_zero_one_distribution.png', dpi=300)


    raw, testing, y = set_df(test_df, FEATURES)
    print(f"----- 사용 모델 -----\n{model}\n")
    result = model.predict_proba(testing)
    raw['pred'] = result[:,1]
    prob_count = test_df.groupby('userID')['answerCode'].count().to_frame('prob_count')
    raw = pd.merge(raw, prob_count, how='left', on = 'userID')
    raw = raw[['userID','test_group','prob_count','total_prob','user_acc','total_used_time','answerCode','pred']]

    zero_one_distribution(raw)
    
    ax = lgb.plot_importance(model);
    ax.figure.savefig('static/_00_lgbm_plot_importance.png')

    threshold = 0.5
    accuracy = accuracy_score(y, (raw['pred'] > threshold).astype('int'))
    roc_auc = roc_auc_score(y, raw['pred'])
    print(f"----- 추론 결과 -----")
    print(f"Accuracy_score : {accuracy*100:.1f} %")
    print(f"ROC_AUC_score : {roc_auc*100:.1f} %")
    data['acc'] = accuracy
    data['roc'] = roc_auc
    return raw, data


