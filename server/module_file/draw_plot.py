import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import lightgbm as lgb
import os
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



def test_group_draw(raw):
    """
        analysis 1 : test_group 별 Prediction 과 Answer 사이의 오차
        plot1 : test_group 별 정답률 bar plot
        plot2 : test_group 별 prediction value

    :param raw: dataframe for analysis
    :return: None
    """

    def test_group_accuracy(raw):
        raw_tgroup = raw.groupby('test_group')['answerCode'].mean()
        plt.figure(figsize=(12, 5))
        plt.bar(raw_tgroup.index, raw_tgroup.values, edgecolor='black')
        for i in range(9):
            plt.text(i - 0.2, raw_tgroup[i] + 0.02, f"{raw_tgroup[i] * 100:.1f}%")
        plt.title('User mean accuracy by Test_group in test_data', loc='left', weight='bold')
        plt.ylim([0, 1])
        plt.suptitle(f"1-1. TEST GROUP ACCURACY", weight='bold', size=20)
        plt.savefig('static/_10_test_group_accrucy.png', dpi=300)


    test_group_accuracy(raw)

    fig, axes = plt.subplots(3, 3, figsize=(20, 12))

    temp = raw[['test_group', 'answerCode', 'pred']]

    for i in range(1, 10):
        temp2 = temp[temp['test_group'] == f'{i}']
        zero = temp2[temp2['answerCode'] == 0].sort_values(by='pred', ascending=False)
        one = temp2[temp2['answerCode'] == 1].sort_values(by='pred', ascending=False)

        axes[(i - 1) // 3][(i - 1) % 3].plot(np.arange(len(zero)), zero['pred'], label='zero prediction', alpha=0.5, color='orange', marker='o')
        axes[(i - 1) // 3][(i - 1) % 3].plot(np.arange(len(one)), one['pred'], label='one prediction', alpha=0.5, color='blue', marker='^')

        axes[(i - 1) // 3][(i - 1) % 3].fill_between(np.arange(len(zero)), zero['pred'], y2=0, alpha=0.2, color='orange')
        axes[(i - 1) // 3][(i - 1) % 3].fill_between(np.arange(len(one)), one['pred'], y2=1, alpha=0.2, color='blue')

        axes[(i - 1) // 3][(i - 1) % 3].set_title(f"test_group : {i}, sample count of group : {len(temp2)}")
        axes[(i - 1) // 3][(i - 1) % 3].axhline(0.5, color='green', linestyle='--')
        axes[(i - 1) // 3][(i - 1) % 3].legend(loc='lower right')
        axes[(i - 1) // 3][(i - 1) % 3].set_ylim([0, 1])

    plt.suptitle(f"1-2. TEST GROUP - PREDICTION", weight='bold', size=20)
    plt.tight_layout()
    plt.savefig('static/_11_prediction_by_test_group.png', dpi=300)




def difference_by_count(raw, count=0):
    """
        analysis 2 : 문제 풀이 수 관련 분석
        plot1 : 문제 풀이 수 관련 prediction 값과 True 값의 분포

    :param raw: dataframe for analysis
    :return: None
    """

    one = raw[(raw['prob_count'] > count) & (raw['answerCode'] == 1)].sort_values(by='pred', ascending=False)
    zero = raw[(raw['prob_count'] > count) & (raw['answerCode'] == 0)].sort_values(by='pred', ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(raw['prob_count'], bins=32, rwidth=0.9, edgecolor='black')
    axes[0].set_xticks(np.arange(15, 1620, 50))
    axes[0].set_xticklabels(np.arange(15, 1620, 50), rotation=45)
    axes[0].set_title("Prob count by user", loc='left', weight='bold')
    axes[0].grid(linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('User count')
    axes[0].set_xlabel('Prob count')

    axes[1].plot(np.arange(len(one)), one['answerCode'], color='green', alpha=0.8)
    axes[1].plot(np.arange(len(one)), one['pred'], color='blue', alpha=0.4, label='one')
    axes[1].fill_between(np.arange(len(one)), one['pred'], y2=1, color='blue', alpha=0.2)

    axes[1].plot(np.arange(len(zero)), zero['answerCode'], color='red', alpha=0.3)
    axes[1].plot(np.arange(len(zero)), zero['pred'], color='orange', alpha=0.4, label='zero')
    axes[1].fill_between(np.arange(len(zero)), zero['pred'], color='orange', alpha=0.2)

    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=0.8)
    axes[1].grid(linestyle='--', linewidth=0.5)

    axes[1].legend(loc='upper right')
    axes[1].set_title(f'Difference between Prediction and True value with prob count over {count}', loc='left', weight='bold')

    plt.suptitle(f"2-1. PROBLEM COUNT - DIFFERENCE", weight='bold', size=20)
    plt.tight_layout()
    plt.savefig('static/_20_difference_by_count.png', dpi=300)



def zero_one_by_count(raw, count=0):
    """
        analysis 2 : 문제 풀이 수 관련 분석
        plot1 : 특정 문제 수 이상을 푼 유저에 한해서, 정답값이 0 과 1 일 때 예측값

    :param raw: dataframe for analysis
    :return: None
    """
    one = raw[(raw['prob_count'] > count) & (raw['answerCode'] == 1)].sort_values(by='pred', ascending=False)
    zero = raw[(raw['prob_count'] > count) & (raw['answerCode'] == 0)].sort_values(by='pred', ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].scatter(np.arange(len(zero)), zero['answerCode'], color='red', alpha=0.3)
    axes[0].scatter(np.arange(len(zero)), zero['pred'], color='orange', alpha=0.3)
    axes[0].bar(np.arange(len(zero)), zero['pred'], color='orange', alpha=0.3, width=0.2)

    axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=0.8)
    axes[0].grid(linestyle='--', linewidth=0.5)
    axes[0].set_title(f'zero prediction value (small area is good) with prob count over {count}', loc='left', weight='bold', size=20)

    axes[1].scatter(np.arange(len(one)), one['answerCode'], color='green', alpha=0.3)
    axes[1].scatter(np.arange(len(one)), one['pred'], color='blue', alpha=0.3)
    axes[1].bar(np.arange(len(one)), one['pred'], color='blue', alpha=0.3, width=0.2)

    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=0.8)
    axes[1].grid(linestyle='--', linewidth=0.5)
    axes[1].set_title(f'one prediction value (big area is good) with prob count over {count}', loc='left', weight='bold', size=20)

    plt.suptitle(f"2-2. PROBLEM COUNT - 0 & 1 DISTRIBUTION", weight='bold', size=20)
    plt.tight_layout()
    plt.savefig('static/_21_prediction_value_by_prob_count.png', dpi=300)





def user_accuracy_prediction(raw, user_acc=0.6, check='upper'):
    """
        analysis 3 : user 별 정답률 관련 분석
        plot1 : 평균 정답률 이상 또는 이하인 유저에 대해서 정답과 예측값 분석

    :param raw: dataframe for analysis
    :return: None
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].hist(raw['user_acc'], bins=50, rwidth=0.85, edgecolor='black')
    axes[0].set_title("user_acc distribution")

    if check == 'upper':
        raw_acc = raw[raw['user_acc'] > user_acc]
    elif check == 'lower':
        raw_acc = raw[raw['user_acc'] < user_acc]

    acc_zero = raw_acc[raw_acc['answerCode'] == 0].sort_values(by='pred')
    axes[1].scatter(np.arange(len(acc_zero)), acc_zero['pred'], alpha=0.5, label='prediction')
    axes[1].scatter(np.arange(len(acc_zero)), acc_zero['answerCode'], alpha=0.5, label='user accuracy')
    axes[1].bar(np.arange(len(acc_zero)), acc_zero['pred'], color='orange', alpha=0.3, width=0.2)
    axes[1].legend(loc='lower right')
    axes[1].set_title(f'zero answer with prediction user accuracy {user_acc} {check}')
    axes[1].axhline(0.5, linestyle='--', color='green')

    acc_one = raw_acc[raw_acc['answerCode'] == 1].sort_values(by='pred')
    axes[2].scatter(np.arange(len(acc_one)), acc_one['pred'], alpha=0.5, label='prediction')
    axes[2].scatter(np.arange(len(acc_one)), acc_one['answerCode'], alpha=0.5, label='user accuracy')
    axes[2].bar(np.arange(len(acc_one)), acc_one['pred'], color='orange', alpha=0.3, width=0.2)
    axes[2].legend(loc='lower right')
    axes[2].set_title(f'one answer with prediction user accuracy {user_acc} {check}')
    axes[2].axhline(0.5, linestyle='--', color='green')

    plt.suptitle(f"3. USER ACCURACY", weight='bold', size=20)
    plt.tight_layout()
    plt.savefig('static/_30_prediction_value_by_user_accuracy.png', dpi=300)






def total_time_with_prediction(raw):
    """
        analysis 4 : 총 문제 풀이 시간에 따른 분석
        plot1 : 특정 총 문제풀이 시간 이상, 이하 를 사용한 유저들에 한하여 정답과 예측값 분석

    :param raw: dataframe for analysis
    :return: None
    """
    fig = plt.figure(figsize=(16, 10))

    ax0 = plt.subplot(3, 1, 1)
    ax1 = plt.subplot(3, 2, 3)
    ax2 = plt.subplot(3, 2, 4)
    ax3 = plt.subplot(3, 2, 5)
    ax4 = plt.subplot(3, 2, 6)

    a = ax0.hist(raw['total_used_time'], bins=40, rwidth=0.8, edgecolor='black')[1][1]
    a_hour = int(a//3600)
    a_minute = int(a%3600/60)
    ax0.set_title("total_user_time distribution")

    raw_upper = raw[raw['total_used_time'] > a]
    raw_lower = raw[raw['total_used_time'] < a]

    acc_zero = raw_upper[raw_upper['answerCode'] == 0].sort_values(by='pred')
    ax1.scatter(np.arange(len(acc_zero)), acc_zero['pred'], alpha=0.5, label='prediction')
    ax1.scatter(np.arange(len(acc_zero)), acc_zero['answerCode'], alpha=0.5, label='user accuracy')
    ax1.bar(np.arange(len(acc_zero)), acc_zero['pred'], color='orange', alpha=0.3, width=0.2)
    ax1.legend(loc='lower right')
    ax1.set_title(f'group with zero answer & upper {a_hour} h {a_minute} m total_used_time')
    ax1.axhline(0.5, linestyle='--', color='green')

    acc_one = raw_upper[raw_upper['answerCode'] == 1].sort_values(by='pred')
    ax2.scatter(np.arange(len(acc_one)), acc_one['pred'], alpha=0.5, label='prediction')
    ax2.scatter(np.arange(len(acc_one)), acc_one['answerCode'], alpha=0.5, label='user accuracy')
    ax2.bar(np.arange(len(acc_one)), acc_one['pred'], color='orange', alpha=0.3, width=0.2)
    ax2.legend(loc='lower right')
    ax2.set_title(f'group with one answer & upper {a_hour} h {a_minute} m total_used_time')
    ax2.axhline(0.5, linestyle='--', color='green')

    acc_zero = raw_lower[raw_lower['answerCode'] == 0].sort_values(by='pred')
    ax3.scatter(np.arange(len(acc_zero)), acc_zero['pred'], alpha=0.5, label='prediction')
    ax3.scatter(np.arange(len(acc_zero)), acc_zero['answerCode'], alpha=0.5, label='user accuracy')
    ax3.bar(np.arange(len(acc_zero)), acc_zero['pred'], color='orange', alpha=0.3, width=0.2)
    ax3.legend(loc='lower right')
    ax3.set_title(f'group with zero answer & lower {a_hour} h {a_minute} m total_used_time')
    ax3.axhline(0.5, linestyle='--', color='green')

    acc_one = raw_lower[raw_lower['answerCode'] == 1].sort_values(by='pred')
    ax4.scatter(np.arange(len(acc_one)), acc_one['pred'], alpha=0.5, label='prediction')
    ax4.scatter(np.arange(len(acc_one)), acc_one['answerCode'], alpha=0.5, label='user accuracy')
    ax4.bar(np.arange(len(acc_one)), acc_one['pred'], color='orange', alpha=0.3, width=0.2)
    ax4.legend(loc='lower right')
    ax4.set_title(f'group with one answer & lower {a_hour} h {a_minute} m total_used_time')
    ax4.axhline(0.5, linestyle='--', color='green')

    plt.suptitle(f"4. TOTAL TIME TO SOLVE", weight='bold', size = 20)
    plt.tight_layout()
    plt.savefig('static/_40_total_time_to_solve')












