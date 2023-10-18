import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
# from PIL import Image

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(
    layout="wide",
)
st.title("⚖️ LGBM ⚖️")
st.subheader(":blue[Overall RMSLE]")
#########################
#결과값
def f1(s1, s2):
    if (s1-s2) < 0:
        delta_color = "inverse"
    else:
        delta_color = "normal"
    value = np.round(abs(s1-s2), 3)  
    return value, delta_color

cv_score = 0.343
valid = 0.416
test = 0.455

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)


col1, col2, col3 = st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)
###############################

#dataset
@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train_eda.csv")
    return train

train= load_data()
columns = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'Day of week',
       'Month', 'Hour', 'Day_info']

st.subheader(":blue[📁 Dataset used in model]")
st.write(train[columns].head(5))

#######################################
#model code
if st.button("⌨️ See code for LGBM with Optuna"):
    st.code("""
    def objective(trial: Trial) -> float:
        params_lgb = {
            "random_state": 42,
            "verbosity": -1,
            "metric": "regression",
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 200, 500),
        }

        n_splits=5
        random_state=2023
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        #local_train, local_valid를 5번 만들어서 수행
        for train_index, valid_index in kf.split(X=x_train, y=y_train):
            X_train, Y_train = x_train.iloc[train_index], np.log(y_train[train_index])
            X_valid, Y_valid = x_train.iloc[valid_index], np.log(y_train[valid_index])

            model = lgbm.LGBMRegressor(**params_lgb)
            model.fit(
                X_train,
                Y_train,
                eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                early_stopping_rounds=100,
                verbose=False
            )

            lgb_pred = model.predict(X_valid)
            scores.append(rmsle(Y_valid, lgb_pred))

        return np.mean(scores)
    """)
####################################
st.subheader(":blue[👍 Best Hyperparameters]")

hp_name = ["n_estimators", "learning_rate", "reg_alpha", "reg_lambda", "max_depth",
            "num_leaves", "colsample_bytree", "subsample", "subsample_freq",
            "min_child_samples", "max_bin"]
types = ["int", "float", "float", "float", "int", "int", "float", "float", "int", "int", "int"]
ranges = ["""[500, 2000]""", """[0.0001, 0.01]""", """"[1e-8, 3e-5]""", """"[1e-8, 9e-2]""",
          """[1, 20]""", """[2, 256]""", """[0.4, 1.0]""", """[0.3, 1.0]""", """[1, 10]""",
          """[5, 100]""", """" [200, 500]"""]
best_values = [1887, 0.009, 2.436, 0.064, 15, 88, 0.74, 0.95, 9, 41, 415]
params = pd.DataFrame({'hp_name':hp_name,
                        "type": types,
                        'range':ranges,
                        'best_value':best_values})
st.write(params)



######################################

op = "bike_prediction/LGBM/lgbm_hour/optuna.png"
hp = "bike_prediction/LGBM/lgbm_hour/hp.png"
result = """bike_prediction/LGBM/lgbm_hour/pred.png"""

st.subheader(":blue[Optuna Result]")
st.markdown("- LGBM의 하이퍼파라미터 중 중요하게 작용했던 것은 learning rate와 n_estimators, max_depth와 subsample임을 확인할 수 있다.")
st.image(op)
st.image(hp)

#######    
st.subheader(":blue[Test data prediction result]")   
st.image(result)

##############################  
st.subheader(":blue[Shap Values]")
sharp = "bike_prediction/LGBM/lgbm_hour/LGBM SHAP.png"
waterfall = "bike_prediction/LGBM/lgbm_hour/LGBM SHAP2.png"

st.markdown("##### Overall Shap Values")
st.warning("- 회색으로 표시된 변수들은 범주형 변수에 해당한다.\n"
           "- 기온이 낮은 경우와 습도가 높은 경우가 count 값을 떨어뜨리는 데 가장 크게 영향을 미쳤음을 확인할 수 있다.\n"
           "- 또한 변수들은 전반적으로 count 값을 증가시키는 방향보다 감소시키는 방향으로 더 많이 작용한다는 것을 알 수 있다.")
st.image(sharp)

st.markdown("##### Model explanation for first row")
st.warning("- 아래는 1건의 개별 데이터에 대한 분석 결과이다.\n"
            "- 이 경우 weather과 day_info, day of week는 count를 증가시키는 방향으로 작용했다.\n"
            "- 그러나 hour, temperature, month, season 등의 feature가 반대 방향으로 더욱 크게 작용하여 낮은 count 값을 예측했음을 알 수 있다.\n"
            "- 영향력의 크기로 보았을 때 전체 변수 중에서는 hour, temperature, month가 중요한 것으로 보인다.")
st.image(waterfall, use_column_width=True)
