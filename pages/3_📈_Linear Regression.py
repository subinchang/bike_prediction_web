import streamlit as st
import numpy as np

import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(
    layout="wide",
)
st.title("📈 Linear Regression 📈")

def f1(s1, s2):
    if (s1-s2) < 0:
        delta_color = "inverse"
    else:
        delta_color = "normal"
    value = np.round(abs(s1-s2), 3)  
    return value, delta_color
#########
st.subheader(":blue[Overall RMSLE]")
###########
st.markdown("##### 1. Ridge")

cv_score = 0.633
valid = 0.617
test = 0.651

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)

col1, col2, col3= st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, -v1, delta_color='inverse')
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)

#########################################################
st.markdown("##### 2. Lasso")

cv_score = 0.634
valid = 0.617
test = 0.646

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)

col1, col2, col3 = st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, -v1, delta_color='inverse')
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)
###############################################################
st.markdown("##### 3. Elastic Net")

cv_score = 0.634
valid = 0.617
test = 0.646

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)

col1, col2, col3= st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, -v1, delta_color='inverse')
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)

st.markdown("")
st.warning('- **CV score:** local train set(1-14일)에 대한 5 fold cross validation score\n'
           '- **Test score:** local test set(15-19일)에 대한 score\n'
           '- **Kaggle score:** Kaggle에 제출한 20일-월말의 target 값(count)에 대한 Out of sample test score')
st.markdown("")

##############################################################
st.title('')
st.subheader(":blue[📁 Dataset used in model]")
st.markdown('- n개의 값을 가진 범주형 변수의 one hot encoding을 진행할 때, **다중공선성 문제를 방지**하기 위해\
            one hot encoding 후 한 col을 drop 해주었다.\n'
            '- 이는 **n-1개의 col**만 있어도 나머지 하나의 col 정보는 자연히 얻을 수 있기 때문이다.')

@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train_eda.csv")
    return train

train= load_data()
cat_col = ["season", "Year","weather", "Day of week","Month","Day_info", "Hour"]
for col in cat_col:
    train[col] = train[col].astype("category")
#target, drop, y
target_col = "count"
drop_cols = ["Unnamed: 0", "datetime", "workingday", "holiday", "Day", "Year", "sin_hour", "cos_hour",target_col]
train = train.drop(drop_cols, axis=1)

cat_col = ["season", "weather", "Day of week", "Month", "Day_info", "Hour"]
train = pd.get_dummies(train, columns=cat_col, drop_first=True)
st.write(train.head(3))

####################################################
st.title('')
st.subheader(":blue[📊 Correlation heatmap]")
fig = px.imshow(train.corr(), text_auto=False,aspect="auto")
fig.update_layout(width = 600,
                 height = 600)
st.plotly_chart(fig, theme=None, use_container_width=False)


##################################################################
st.title('')
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📄 Ridge Hyperparameters")

    hp_name = ["scalers", "dim_red", "pca_n_components", "alpha"]
    types = ["scaler", "categorical", "int", "float"]
    ranges = [""" ['minmax', 'standard', 'robust']""",
            """[PCA, NONE]""",
            """[2, x_train.shape[1]]""",
            """[0, 10]"""]
    best_values = ['standard', None, None, 2.385]
    params = pd.DataFrame({'hp_name':hp_name,
                            "type": types,
                            'range':ranges,
                            'best_value':best_values})
    st.write(params)

with col2:
    st.markdown("##### 📄 Lasso Hyperparameters")
    hp_name = ["scalers", "dim_red", "pca_n_components", "alpha"]
    types = ["scaler", "categorical", "int", "float"]
    ranges = [""" ['minmax', 'standard', 'robust']""",
            """[PCA, NONE]""",
            """[2, x_train.shape[1]]""",
            """[0001, 1]"""]
    best_values = ['standard', None, None, 0.001]
    params = pd.DataFrame({'hp_name':hp_name,
                            "type": types,
                            'range':ranges,
                            'best_value':best_values})
    st.write(params)

with col3:
    st.markdown("##### 📄 Elastic Net Hyperparameters")
    hp_name = ["scalers", "dim_red", "pca_n_components", "alpha", "l1_ratio"]
    types = ["scaler", "categorical", "int", "float", "float"]
    ranges = [""" ['minmax', 'standard', 'robust']""",
            """[PCA, NONE]""",
            """[2, x_train.shape[1]]""",
            """[0001, 1]""",
            """[0, 1]"""]
    best_values = ['standard', None, None, 0.001, 0.505]
    params = pd.DataFrame({'hp_name':hp_name,
                            "type": types,
                            'range':ranges,
                            'best_value':best_values})
    st.write(params)

###################################################################
@st.cache_data
def load_data():
    coeff = pd.read_csv("bike_prediction/Linear/회귀계수_final.csv",index_col=0)
    return coeff

coef = load_data()

##################################################################
st.subheader(":blue[🏷️ Coefficients]")
colors = ['blue', 'green', 'red']

st.write(coef.T)
# 플롯 생성
fig = go.Figure()

#산점도 그리기
for i, col in enumerate(coef.columns):
    fig.add_trace(go.Scatter(
        x=coef.index,
        y=coef[col],
        mode='markers',
        name=col,
        marker=dict(
            color=colors[i-1],
            size=6,
        ),
        
    ))

# 가운데 x축과 평행한 선 그리기
fig.add_shape(type="line",
              x0=0,
              y0=0,
              x1=52,
              y1=0,
              line=dict(
                  color="black",
                  width=2,
                  dash="dashdot",
              ))

fig.update_layout(
    title="Lasso, Ridge, Elastic Net Regression Coefficients",
    xaxis_title="변수",
    yaxis_title="회귀계수 값",
    legend=dict(
        x=-0.15,
        y=1,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)',
    ),
    width = 1200,
    height = 800,
    xaxis=dict(
        tickangle=300, # 300도회전.
        tickfont=dict(size=10) # x축 레이블의 폰트 크기를 조정합니다.
    )
)

# 그래프 출력
st.plotly_chart(fig, theme=None, use_container_width=True)

# 해석
st.markdown("- 선형 모델의 경우 Hour와 Season 변수가 count 값에 영향을 많이 미치는 것으로 나타났다.\n"
            "- 또한, Ridge는 계수를 0에 근사하도록 축소하나, Lasso는 계수를 완전히 0으로 만든다는 점에서 가장 큰 차이를 보였다.")

