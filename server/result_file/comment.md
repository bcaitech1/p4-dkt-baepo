## 전체 구조
* Model inference
    0) Feature importance
    1) prediction histogram distribution by 0 & 1
    
* analysis 1 : test_group 별 결과 체크
    0) test_group 별 정답 값과 예측 값의 오차
    
* analysis 2 : 문제풀이 수 별 결과 체크 
    0) 문제풀이 수 컬럼 관련 Prediction 값과 True 값의 차이 분포
    1) 특정 문제 수 이상을 푼 유저에 한해서, 정답값이 0 과 1 일 때 prediction 분포
    
* analysis 3 : 각 유저의 정확도 에 따른 결과 체크
    0) 각 유저의 정확도 분포 및 Prediction 값과 정답 값 사이의 오차
    
* analysis 4 : 문제 풀이 총 시간에 따른 결과 체크
    0) 문제풀이 총 시간 및 정답값에 따른 Prediction 분포
  
----


### 00 모델의 Feature importance
* 영향을 가장 많이 준 컬럼은 [total_prob, user_acc, total_used_time] 세 가지 이다.
* total_prob : 총 문제풀이 수 에 관련된 컬럼
* user_acc : 유저가 푼 문제의 정답률
* total_used_time : 유저가 문제푸는데 사용한 총 시간

### 01 모델의 예측 결과 확률값 분포
* 0을 예측하는 확률 값은 0에 가깝게, 1을 예측하는 확률 값은 1에 가깝게 분포가 분리되어 형성될 수록 좋은 예측이라고 볼 수 있다.
* 전반적으로 1을 더 잘 맞추는 쪽으로 분포가 잘 형성된 것을 볼 수 있다.

유저별 문제를 푼 종류, 문제 수 등이 다르기 때문에 상위 3개 column 에 대해서 시각화해보자.



### 11 test_group 별 Prediction 과 Answer 사이의 오차
그래프의 면적이 작을수록 각 label 에 해당하는 Prediction 이 잘 이루어졌다고 볼 수 있다.  
전반적으로 test_group number 가 올라갈수록, 0 에 대한 prediction 오차가 줄어들고, 1 에 대한 prediction 오차가 늘어나는 것을 볼 수 있다.  
이는, test_group number 가 올라갈수록, 정답률이 떨어지기 때문에 0에 대한 데이터가 1에 대한 데이터보다 많아 학습이 잘 되어 나타나는 결과로 보인다.



### 20 문제풀이 수 컬럼 관련 Prediction 값과 True 값의 차이 분포
* 문제풀이 수 에 해당하는 조건을 더 주고 Prediction 값과 True 값 사이의 오차가 어떻게 변하는지 알아본다.

### 21 특정 문제 수 이상을 푼 유저에 한해서, 정답값이 0 과 1 일 때 예측값
* 특정 문제 수 이상을 푼 유저에 한해서, 정답값이 0 과 1 일 때 예측을 잘 하는지 분포를 알아보자.



### 30 각 유저의 정확도 분포 및 Prediction 값과 정답 사이의 오차
* 특정 정확도 이상 혹은 이하 유저에 한해서 Prediction 값의 분포가 어떻게 변하는지 알아보자.



### 40 문제풀이 총 시간 및 정답값에 따른 Prediction 분포
* 문제 풀이 총 시간에 따른 User histogram 분포
* 특정 문제풀이 시간 이상 투자한 그룹과 이하 투자한 그룹별 Prediction 값과 정답값의 오차.