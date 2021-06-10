import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from enum import Enum
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class PreprocessArgs:
    asset_dir: str
    data_dir: str
    n_questions: int = 0
    n_test: int = 0
    n_tag: int = 0


@dataclass
class DatasetArgs:
    num_workers: int
    batch_size: int
    max_seq_len: int


class Preprocess:
    def __init__(self, args: PreprocessArgs):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data, {
            'n_questions': self.args.n_questions,
            'n_test': self.args.n_test,
            'n_tag': self.args.n_tag,
        }

    def get_test_data(self):
        return self.test_data, {
            'n_questions': self.args.n_questions,
            'n_test': self.args.n_test,
            'n_tag': self.args.n_tag,
        }

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):     # sklearn.LabelEncoder를 사용해서 모든 범주형 변수를 숫자로 변경
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            
            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']  
                # unknown 값은 test 셋에서 없는 값이 나왔을 대 처리하기 위함
                le.fit(a)
                self.__save_labels(le, col)
                # Train에서는 le를 새로 만들고 저장
            else:
                # Test에서는 저장한 le를 불러오기
                label_path = os.path.join(self.args.asset_dir, col + '_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        #TODO
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        
        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values,
                    r['userID'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args: DatasetArgs):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]
        
        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        # 뒤쪽에서부터
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args: DatasetArgs, train, valid):

    pin_memory = False
    train_loader: DataLoader = None
    valid_loader: DataLoader = None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader


### Custom Code ###

class FeatureType(Enum):
    Categorical = 0
    Continuous = 1

@dataclass()
class FeatureInfo():
    name: str
    feature_type: FeatureType
    num_class: int = -1

class NewDKTDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: DataFrame,
            data_feature_info: Dict[str, FeatureInfo],
            max_seq_len: int,
            seed: int,
            target_features: List[str],
            shake_range: int
        ) -> None:
        user_groups = data.groupby('userID')
        self.df_raw: List[DataFrame] = [user_group for _ , user_group in user_groups] 
        self.max_seq_len = max_seq_len
        self.random = random.Random(seed) if seed else random.Random()
        # Epoch가 반복되어도 같은 데이터를 불러오지 않음
        self.features: List[FeatureInfo] = []
        for feature in target_features:
            self.features.append(data_feature_info[feature])
        self.shake_range = shake_range
        

    def __getitem__(self, index):
        # user 별 데이터 반환
        user_df = self.df_raw[index]

        result = []

        if len(user_df) > self.max_seq_len:     # 길면 자르기
            if len(user_df) > self.max_seq_len + self.shake_range:
                end_index = len(user_df) - self.random.randint(0, self.shake_range)
                start_index = end_index - self.max_seq_len
            else:
                end_index = len(user_df)
                start_index = end_index - self.max_seq_len
            target_df: DataFrame = user_df.iloc[start_index:end_index]
        else:
            target_df: DataFrame = user_df

        # 반환 값 생성
        for target_feature in self.features:
            result.append(
                torch.tensor(
                    target_df[target_feature.name].to_numpy(na_value=0),
                    dtype=torch.long if target_feature.feature_type == FeatureType.Categorical else torch.float
                )
            )
        result.append(
            torch.tensor(target_df['answerCode'].to_numpy())
        )
        mask = np.zeros(self.max_seq_len, dtype=np.int16)   # mask 생성
        mask[-len(user_df):] = 1
        result.append(torch.tensor(mask))
        # result.append(
        #     torch.tensor(target_df['userID'].to_numpy())
        # )   # 디버깅용, 사용되지 않음
        # [*[Features], answerCode, mask, userID]
                
        return result
    
    def __len__(self):      # user 수
        return len(self.df_raw)
    
    def collate(self, batch: List[List[torch.Tensor]]):
        col_n = len(batch[0])
        col_list = [[] for _ in range(col_n)]

        # batch의 값들을 각 column끼리 그룹화
        for row in batch:
            for i, col in enumerate(row):
                pre_padded = torch.zeros(self.max_seq_len, dtype=col.dtype)
                pre_padded[-len(col):] = col
                col_list[i].append(pre_padded)

        for i, _ in enumerate(col_list):
            col_list[i] = torch.stack(col_list[i])
        
        return tuple(col_list)

    
def generate_dataset(
        data_path: str,
        asset_dir: str,
        valid_split: bool,
        gss_args: Dict,
        dataset_args: Dict,
    ) -> List[Tuple[NewDKTDataset, NewDKTDataset]]:
    dtype = {
        'userID': 'int16',
        'answerCode': 'int8',
        'KnowledgeTag': 'int16'
    }   
    df = pd.read_csv(data_path, dtype=dtype, parse_dates=['Timestamp'])
    df, feature_info = __process_feature(df, asset_dir)

    result = []

    if valid_split:
        gss = GroupShuffleSplit(**gss_args)
        # 마지막 시험의 정답을 기준으로 분류 (Office Hour에서 언급)
        for index, (train_idx, valid_idx) in enumerate(gss.split(X=df, y=df['last_question_group'], groups=df['userID'])):
            logger.info(f'Generating dataset fold {index}...')
            train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]

            result.append((
                NewDKTDataset(train_df, feature_info, **dataset_args), 
                NewDKTDataset(valid_df, feature_info, **dataset_args)
            ))
        logger.info(f'Spliting dataset finished')
    else:
        result.append((NewDKTDataset(df, feature_info, **dataset_args), None))

    return result

def __process_feature(
        df: DataFrame, 
        encoder_save_dir: str
    ) -> Tuple[DataFrame, Dict[str, FeatureInfo]]:
    """Feature Engineering 수행 및 범주형 데이터 부호화

    Args:
        df (DataFrame): 데이터 원본

    Returns:
        DataFrame: 변형된 데이터
    """
    # pandas.apply와 pandas.transform의 차이
    # https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object
    logger.info('Processing...')
    logger.info('Converting timestamp to int...')
    df['Time'] = df['Timestamp'].apply(lambda x: int(x.asm8) // 1000000000)
    # Baseline과 다른데 이 코드가 훨씬 빠르고 정확하다. 
    # Baseline에서는 timezone마다 값이 다르게 출력된다. 
    # 이것은 하나의 컴퓨터에서는 문제가 없지만 해당 모델을 다른 컴퓨터로 가져가면 제대로 동작하지 않을 수 있다.
    # 이 코드에서 시간대는 무조건 UTC 기준으로 변환된다. (Timezone 정보가 없이 처리되므로....)
    # 물론, 데이터의 시간대는 KST 기준이겠지만 시간 흐름이 중요하므로 별도 처리하지는 않았다.
    logger.info('Converting assessmentItemID to features...')
    df['assessment_group'] = df['assessmentItemID'].apply(lambda x: int(x[2:3]))
    df['assessment'] = df['assessmentItemID'].apply(lambda x: int(x[4:7]))
    df['question'] = df['assessmentItemID'].apply(lambda x: int(x[7:]))

    logger.info('Converting Time to features...')
    df['test_interval'] = df['Time'] - df['Time'].shift(1, fill_value=0)   # 시간 간격
    # fill_value를 쓰지 않으면 int형식으로 저장되지 않는다.
    df.loc[df['testId'] != df['testId'].shift(1), ('test_interval')] = 0
    # DataFrame에 직접 값을 대입하면 pandas에서 SettingWithCopyWarning을 발생시킨다.
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    # 요는, 직접적으로 값을 바꾸는 행위를 지양한다는 이야기

    df['test_interval_sum'] = df.groupby(['testId'])['test_interval'].cumsum()
    df['user_interval_sum'] = df.groupby(['userID'])['test_interval'].cumsum()

    logger.info('Add features related accuracy...')
    correct_answer = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    total_question = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] =  correct_answer / total_question
    # 정답률 = 이전까지 맞춘 정답 수 / 총 문제 풀이 수
    df['interaction'] = df['answerCode'].shift(1, fill_value=0).add(1)   # Interaction 이전 정답 여부
    df.loc[df['testId'] != df['testId'].shift(1), ('interaction')] = 0
    
    # correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])         # test 시험지 별 정답률 평균, 총합 
    # correct_t.columns = ["test_mean", 'test_sum']
    # correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])   # 문제 tag 별 정답률 평균, 총합
    # correct_k.columns = ["tag_mean", 'tag_sum']
    # 위 변수들은 현재는 필요 없다고 판단.

    # Stratify를 위한 마지막 문제의 정답 그룹
    logger.info('Add features last question answer group...')
    df['last_question_group'] = df.groupby(['userID'])['answerCode'].transform(lambda x: x.iloc[-1])

    # Feature Engineering 결과 확인
    # df.to_csv('temp.csv')

    # 범주형 데이터 인코딩
    logger.info('Encoding categorical features...')
    save_path = os.path.join(encoder_save_dir, 'encoders.npy')
    all_cols = [    # Feature 추가 때마다 수정할 것
        'KnowledgeTag', 
        'Time', 
        'assessment_group', 
        'assessment', 
        'question', 
        'test_interval', 
        'test_interval_sum', 
        'user_interval_sum', 
        'user_acc',
        'interaction',
    ]
    cate_cols = [
        'KnowledgeTag', 
        'assessment_group', 
        'assessment', 
        'question',
        'interaction',
    ]  # 여기에 범주형 지정, 마찬가지로 추가 때마다 수정할 것
    encoders: Dict[str, LabelEncoder] = {}

    if os.path.isfile(save_path):   # 이미 파일이 존재하는 경우 (Feature Engineering 변화가 없는 경우)
        with open(save_path, 'rb') as fr:
            encoders = pickle.load(fr)
        logger.info('Searching and converting unknown value...')
        try:
            for col in cate_cols:
                df[col] = df[col].apply(lambda x: x if str(x) in encoders[col].classes_ else 'unknown')

            logger.info('Checking unknown value...')
            unk_count = 0
            for col in cate_cols:
                for data in df[col]:
                    if data == 'unknown':
                        unk_count = unk_count + 1
            if unk_count > 0:
                logger.info(f'{unk_count} Unknown token found')
            # 검사 과정이 새로 만드는 것보다 시간이 더 걸림. 수정 방안 고민해보기
        except:
            raise Exception('Loaded encoders are not match with defined categorical columns')

    else:       # 새로 만들 경우
        for col in cate_cols:
            label_encoder = LabelEncoder()

            col_values = df[col].unique().tolist() + ['unknown']
            label_encoder.fit(col_values)

            encoders[col] = label_encoder
        with open(save_path, 'wb') as fw:
            pickle.dump(encoders, fw)

    for col in cate_cols:
        df[col]= df[col].astype(str)
        test = encoders[col].transform(df[col])
        df[col] = test
        # 0이 문제라면 여기서 1을 더해주면 안 되나?

    feature_info: Dict[str, FeatureInfo] = {}
    for feature in all_cols:
        if feature in cate_cols:
            feature_info[feature] = FeatureInfo(
                feature,
                FeatureType.Categorical,
                len(encoders[feature].classes_)
            )
        else:
            feature_info[feature] = FeatureInfo(
                feature,
                FeatureType.Continuous,
            )
    logger.info('Processing Complete')
    return df, feature_info