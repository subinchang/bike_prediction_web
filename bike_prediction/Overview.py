import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# 페이지 기본 설정
st.set_page_config(
    layout="wide",
)

# Header
st.title("🚴 Bike Sharing Demand Prediction Analysis 🚴")

# Overview
st.markdown('#### :blue[1. Overview]')
st.markdown('- 제공된 데이터: 2011~12년 동안의 날짜 및 시간대별 자전거 대여 건수\n'
            '- 사용한 모델: Linear(Lasso, Ridge, Elastic Net), KNN, LGBM, CatBoost, Random Forest, XGBoost, TabNet Regression\n'
            '- Kaggle에서 제공된 데이터는 train set이 1일-19일, test set이 20일-말일로 날짜에 따라 구분되어 있다.\n'
            '- 따라서 Kaggle 기준에 맞춰 20일-말일의 시간대별 자전거 대여 건수를 예측하였고, 자동 채점되는 점수로 성능을 평가하였다.')

# Dataset Information
@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train.csv")
    test = pd.read_csv("bike_prediction/kaggle_data/test.csv")
    return train, test
train, test = load_data()

st.markdown('#### :blue[2. Dataset Information]')

col1, col2 = st.columns([1,5])

def color_coding(row):
    if row.attributes in ["season", 'workingday', 'weather', 'atemp', 'casual', 'registered']:
        return ['background-color:pink'] * len(row)
    else:
        return [None] * len(row)



with col1:
   cols = pd.DataFrame(train.columns)
   cols.columns = ['attributes']
   st.dataframe(cols.style.apply(color_coding, axis=1), width=None, height=460)

with col2:
   st.write('')
   st.markdown("중요 변수 및 해석에 유의해야 하는 변수는 다음과 같다.")
   st.markdown('- **season**\n'
               '    - 1-3월, 4-6월, 7-9월, 10-12월이 각각 1, 2, 3, 4로 할당되어 있다.\n'
               '   - Kaggle에는 1=봄, 2=여름, 3=가을, 4=겨울이라고 되어 있으나 실제로는 그렇게 볼 수 없으므로 1-4의 구분을 그대로 사용하였다.\n'
            '- **workingday**\n'
            '   - 주말도 공휴일도 아닌 날\n'
            '- **weather**\n'
            '   - 1 > 2 > 3 > 4 순서로 날씨가 좋음을 의미한다.\n'
            '- **atemp**\n'
            '   - 실제 기온이 아닌 체감 온도를 의미한다.\n'
            '- **casual, registered**\n'
            '   - 각각 비회원의 자전거 대여 건수와 회원의 자전거 대여 건수를 의미한다.\n'
            '   - casual + registered = count 임을 확인하였다.')

# Evaluation
st.markdown('#### :blue[3. Evaluation]')
st.markdown('- 성능 평가 지표로 Kaggle에서 제시한 RMSLE (Root Mean Squared Logarithmic Error)를 사용하였다. 식은 다음과 같다.')
st.markdown('''
<style>
.katex-html {
    text-align: left;
}
</style>''',
unsafe_allow_html=True
)
st.latex(r'''
RMSLE\text{ }= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(log(p_{i}+1)-log(a_{i}+1))^{2}}
''')
st.markdown('- RMSLE는 회귀 문제에서 모델의 예측값(p)과 실제값(a)의 차이를 측정하는 지표 중 하나이다.\n'
            '- RMSLE는 RMSE와 유사하지만, 로그 변환을 수행하기 때문에 값의 크기 차이에 덜 민감하며, 데이터셋이 정규분포를 따르지 않을 때에도 유용하다는 특징이 있다.\n'
            '- 실제로 주어진 데이터셋 또한 Left-Skewed한 모습을 보여, RMSLE의 사용이 적절하다고 보았다.\n')
# Goals
st.markdown('')
st.markdown('#### :blue[4. Goals]')
st.markdown('- 1차적인 분석 목표는 제시된 baseline code를 통해 산출되는 RMSLE(1.018)보다 나은 결과를 도출하는 것으로 설정하였다.\n'
            '- 더 나아가 2차적으로는 Kaggle Leaderboard에서도 상위권의 점수를 기록하는 것을 목표로 하였다.')