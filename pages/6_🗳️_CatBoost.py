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

st.set_page_config(
    layout="wide",
)
##################
#image import
# def load_image(path):
#     image = Image.open(path)
#     return image
lgb_model = "LGBM\lgbm_log\model.png"
op = "LGBM\lgbm_log\optuna.png"
hp = "LGBM\lgbm_log\importance.png"
result = """LGBM\lgbm_log\pred.png"""


st.title("🗳️ Catboost 🗳️")
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

cv_score = 0.354
valid = 0.362
test = 0.447

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)


col1, col2, col3 = st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("kaggle(out of sample)", test, v2, delta_color=d2)
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
st.write(train[columns].head(3))

#######################################
#model code
if st.button("⌨️ See code for Catboost with Optuna"):
    st.code("""
    def objective(trial):
        cat_param = {
            'iterations':trial.suggest_int("iterations", 500, 2000),
            'od_wait':trial.suggest_int('od_wait', 500, 2300),
            'learning_rate' : trial.suggest_float('learning_rate',0.001, 1),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-2,100),
            'subsample': trial.suggest_float('subsample',0,1),
            'random_strength': trial.suggest_float('random_strength',10,50),
            'depth': trial.suggest_int('depth',1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
            'bagging_temperature' :trial.suggest_float('bagging_temperature', 0.01, 100.00),
            'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "used_ram_limit": "8gb",
            'cat_features': cat_col,
        }
        
        n_splits=5
        random_state=42
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_index, valid_index in kf.split(X=x_train, y=y_train):
            X_train, Y_train = x_train.iloc[train_index], y_train[train_index]
            X_valid, Y_valid = x_train.iloc[valid_index], y_train[valid_index]

            model = CatBoostRegressor(**cat_param)
            model.fit(X_train, np.log1p(Y_train),verbose=0)

            xg_pred = model.predict(X_valid)
            scores.append(rmsle(np.log1p(Y_valid), xg_pred))
        
        return np.mean(scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="catboost_parameter_opt",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=50)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    """)
####################################
st.subheader(":blue[👍 Best Hyperparameters]")

hp_name = ["iterations", 'od_wait', 'learning_rate', 'reg_lambda',
          'subsample', 'random_strength', 'depth', 'min_data_in_leaf',
          'leaf_estimation_iterations', 'bagging_temperature', 'colsample_bylevel',
          "boosting_type"]
types = ["int", "int", "float", "float", "float", "float", "int", "int", "int",
        "float", "float", "categorical"]
ranges = ["""[500, 2000]""", """[500, 2300]""", """[0.001, 1]""", """[1e-2,100]""",
         """[0, 1]""", """[10,50]""", """[1, 15]""", """[1,30]""", """[1,15]""",
         """[0.01, 100.00]""", """[0.4, 1.0]""", """["Ordered", "Plain"]"""]
best_values = [1674, 1265, 0.0358, 72.228, 0.324, 12.316, 8, 18, 14, 49.470,
              0.908, 'Plain']
params = pd.DataFrame({'hp_name':hp_name,
                        "type": types,
                        'range':ranges,
                        'best_value':best_values})
st.write(params)

######################################
st.subheader(":blue[Optuna Result]")

op="bike_prediction/Catboost/op.png"
hp = "bike_prediction/Catboost/hp.png"
result = "bike_prediction/Catboost/pred.png"

st.image(op)
st.image(hp)

st.subheader(":blue[Test data prediction result]")
st.warning("- 이전 모델들에 비해 Catboost의 예측 결과가 덜 분산된 것으로 나타났다.")
st.image(result)

##############################################
st.subheader(":blue[Shap Values]")
shap = "bike_prediction/Catboost/cat_shap.png"
waterfall = "bike_prediction/Catboost/cat_shap2.png"

st.markdown("##### Overall Shap Values")
st.warning("- LGBM과 마찬가지로, 개별 변수의 측면에서는 기온이 낮거나 습도가 높거나 풍속이 높을 때 대여 수를 낮게 예측했음을 알 수 있다.\n"
           "- 위와 같은 경우는 기온이 높거나 습도가 낮거나 풍속이 낮은, 즉 변수가 반대로 작용했을 때보다 더욱 영향력이 큰 것을 확인할 수 있다.")
st.image(shap)

st.markdown("##### Model explanation for first row")
st.warning("- 해당 건의 데이터의 경우 temperature, hour, season, day_info, month가 가장 크게 영향을 미쳤고, 그 중 day_info만 count를 증가시키는 방향으로 작용했다.\n"
          "- 동시에 humidity, windspeed, day of week, weather는 그다지 중요하지 않았던 것으로 보인다.")
st.image(waterfall)
