import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

st.set_page_config(
    layout="wide",
)
st.title("🚴 Conclusion 🚴")
st.subheader(":blue[Our Best Model]")
st.markdown("#### 🌳 Random Forest 🌳")
def f1(s1, s2):
    if (s1-s2) < 0:
        delta_color = "inverse"
    else:
        delta_color = "normal"
    value = np.round(abs(s1-s2), 3)  
    return value, delta_color

cv_score = 0.381
valid = 0.420
test = 0.442

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)

col1, col2, col3 = st.columns(3)
col1.metric("CV_score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)

st.subheader(":blue[Kaggle Leaderboard에서는?]")
import os

rank = "bike_prediction/Conclusion/leaderboard.png"
year = "bike_prediction/Conclusion/using_year.png"

st.markdown("##### 실제 리더보드 순위 점수")
st.image(rank)

st.title('')
st.markdown("##### 최상위권 코드 살펴보기")
if st.button("📁 Using Year"):
    st.code("""
            # parse datetime colum & add new time related columns
        dt = pd.DatetimeIndex(df['datetime'])
        df.set_index(dt, inplace=True)

        df['date'] = dt.date
        df['day'] = dt.day
        df['month'] = dt.month
        df['year'] = dt.year
        df['hour'] = dt.hour
        df['dow'] = dt.dayofweek
        df['woy'] = dt.weekofyear
    """)

st.markdown('- Leaderboard의 상위권인 코드 파일을 살펴본 결과, Year를 변수로 사용한 것을 알 수 있었다.\n'
            '- 대회의 평가 지표인 RMSLE는 **예측값이 실제값보다 작을 경우** penalty를 준다.')
st.warning('- 이는 운영 측면에서 보았을 때, 초과공급이 초과수요보다 낫다고 판단한 것으로 볼 수 있다.\n'
            '- 즉, **수요가 있음에도 대여하지 못하는 경우가 발생하는 것보다는 차라리 자전거를 충분히 준비해두는 것을 선호**한 것이라고 해석하였다.')
st.markdown('- 아래 2개의 prediction plot을 보면 Year를 사용한 경우와 그렇지 않은 경우가 어떻게 모델 평가 지표에 영향을 주는지 알 수 있다.')


# sharp = "XGB/xgb shap.png"
# waterfall = "XGB/xgb shap2.png"
# ### 맥이랑 window랑 백슬래쉬 반대키!!!
# st.markdown("Overall Shap Values")
# st.image(sharp)


# def load_image(path):
#     image = Image.open(path)
#     return image

im1 = "bike_prediction/LGBM/lgbm_log/pred.png"
im2 = "bike_prediction/LGBM/lgbm_year/pred.png"
# image1 = load_image(im1)
# image2 = load_image(im2)

# st.image([image1,image2])
########################

# image1 = io.imread(im1)
# image2 = io.imread(im2)
# fig = make_subplots(1, 2)
# We use go.Image because subplots require traces, whereas px functions return a figure
# fig.add_trace(go.Image(z=image1), 1, 1)
# fig.add_trace(go.Image(z=image2), 1, 2)
# fig.update_layout()
# st.plotly_chart(fig, theme=None, use_container_width=True)

#####################
#💡이 방식이 제일 깔끔한거 같
col1, col2 = st.columns(2)

with col1:
    st.subheader("without Year")
    st.image(im1)

with col2:
    st.subheader("with Year")
    st.image(im2)
##############

st.subheader(":blue[그렇다면 우리는 왜 Year 변수를 사용하지 않았는가?]")
st.markdown('- Year 변수는 예측하고자 하는 기간이 **2011, 2012년에 한정되는 경우에만 유의미**하다.\n'
            '- 따라서 이 변수는 미래의 임의의 시점에 대해 대여량을 예측할 때 유용한 정보를 제공하지 않기 때문에 성능이 떨어지더라도 이를 감수하고 Year를 사용하지 않았다.\n'
            '- 반면 날씨, 습도, 공휴일 등과 같은 변수들은 연도에 관계없이 대여 건수를 예측하는 데 중요한 영향을 미칠 수 있기 떄문에 해당 변수들에 집중하여 모델링을 진행하였다.')

