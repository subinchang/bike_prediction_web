import streamlit as st
import pandas as pd

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# 페이지 기본 설정
st.set_page_config(
    layout="wide",
)
st.title("🧬 Workflow 🧬")


st.warning("**모든 모델에 공통적으로 적용되는 Workflow는 다음과 같다.**\n"
            "1. train set을 train set과 validation set으로 나누기\n"
            "2. train set을 5 fold로 나누고 Cross Validation과 Optuna를 통해 하이퍼파라미터 튜닝\n"
            "3. Optuna에서 도출된 Best Parameter를 이용해 train set을 학습하고 validation set에 대해 성능 확인\n"
            "4. target 값(count)이 없는 전체 test set에 대해 동일한 모델을 학습시킨 후 예측한 결과를 Kaggle에 제출하여 점수 확인")

st.subheader(":blue[1. train_test_split]")
st.markdown('- 매월 20일부터 월말은 target하는 대여 건수 값인 count가 비어 있는 test data이다.\n'
            '- 따라서 1일부터 14일까지를 **local train set**으로, 15일부터 19일까지를 **local test set**으로 설정하였다.')
@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train_eda.csv")
    test = pd.read_csv("bike_prediction/kaggle_data/test_eda.csv")
    return train, test

train, test = load_data()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.Day.unique(), y=[30]*len(train.Hour.unique()), mode="markers", name="train"))
fig.add_trace(go.Scatter(x=list(range(1, 15)), y=[20]*len(list(range(1, 15))), mode="markers", name="local_train"))
fig.add_trace(go.Scatter(x=list(range(15, 20)), y=[20]*len(list(range(15, 20))), mode="markers", name="local_test"))
fig.add_trace(go.Scatter(x=test.Day.unique(), y=[10]*len(test.Hour.unique()), mode="markers", name="test"))

fig.update_layout(title=dict({"text": """Days in test and train"""}))
fig.update_layout(yaxis=dict({"tickvals": [10, 30],
                            "ticktext": ["test", "train"]}))
fig.update_layout(xaxis=dict({"title": "Days"}))

st.plotly_chart(fig, theme=None, use_container_width=True)
st.markdown("- 데이터를 구분한 뒤, local_train을 5 fold cross validation을 통해 하이퍼파라미터를 최적화 하였으며 이때 Optuna를 사용하였다.\n"
            '- 이후 Optuna로 찾은 하이퍼파라미터를 대입하여 local_train 전체를 학습시켰고, local_test의 score를 확인하였다.\n'
            '- 다시 전체 train set에 동일한 모델을 학습시켰고, test set에 대한 예측값을 바탕으로 Kaggle에 제출하여 Kaggle score를 확인하였다.')


st.subheader('')
st.subheader(":blue[2. Custom metric]")
st.markdown('모델의 성능 평가 지표가 RMSLE인 만큼 초기에는 cross validation을 통한 하이퍼파라미터 튜닝 과정에서 scoring을 RMSLE로 사용하였다..\n'
            '그러나 예측값이 음수일 경우 log(y_pred+1)의 값을 계산할 수 없어 오류가 발생하는 것을 알 수 있었다.\n'
            '따라서 이를 방지하기 위해 각 fold에서 학습을 수행할 때 log 변환을 해준 y값을 사용하고 RMSLE 함수를 따로 만들어서 사용해주었다.')
st.code("""
def rmsle(y_true, y_pred, convertExp=True):
    # 지수변환
    if convertExp:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        
    # 로그변환 후 결측값을 0으로 변환
    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))
    
    # RMSLE 계산
    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output
""")
st.code("""
# sklearn을 사용할때 custom objective function을 사용할 경우 make_scorer을 통해서 만들어 주어야 한다
rmsle_scorer = make_scorer(rmsle, greater_is_better = False)

    # local_train, local_valid를 5번 만들어서 수행
    for train_index, valid_index in kf.split(X=x_train, y=y_train):
        X_train, Y_train = x_train.iloc[train_index], np.log1p(y_train[train_index])
        X_valid, Y_valid = x_train.iloc[valid_index], np.log1p(y_train[valid_index])

        model = LGBMRegressor(**params_rf)
        model.fit(X_train, Y_train )

        rf_pred = model.predict(X_valid)
        scores.append(rmsle(Y_valid, rf_pred))
""")
st.markdown('##### 이때 np.log(y)가 아닌 np.log1p(y)를 사용한 이유는?')
st.markdown("log함수의 특성상 y=0이면 $(-\infty)$가 되기 때문에, log(y+1)로 계산되어 y=0일때 0을 반환하는 np.log1p를 사용하였다.")


st.title('')
st.subheader(":blue[3. Pycaret]")
st.markdown('- 모델을 하나씩 돌려보기 전에, AutoML이 가능한 파이썬 라이브러리인 Pycaret을 통해 전반적인 모델의 성능을 파악하고 큰 그림을 그려보았다.\n'
           '- 각 지표에서 가장 점수가 높은 노란색 부분을 보면 트리 기반 모델이 상위권임을 알 수 있다.')
pycaret = "bike_prediction/Pycaret/파이캐럿 결과.png"
st.image(pycaret)


st.title('')
st.subheader(":blue[4. 모델 해석]")
st.markdown("- 일반적으로 해석하기 어려운 트리 기반의 ensemble, boosting 알고리즘은 SHAP를 이용해 모델을 이해하고자 하였다.\n"
            '- SHAP은 블랙박스 모델 등에 대해 feature importance를 파악하게 해주는 방식이다.')
##############
