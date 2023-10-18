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
st.title("👥 KNN Regression 👥")
st.subheader(":blue[Overall RMSLE]")

#####################################################
def f1(s1, s2):
    if (s1-s2) < 0:
        delta_color = "inverse"
    else:
        delta_color = "normal"
    value = np.round(abs(s1-s2), 3)  
    return value, delta_color

cv_score = 0.276
valid = 0.469
test = 0.5571

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)


col1, col2, col3 = st.columns(3)
col1.metric("CV_score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)
###############################


#### 데이터셋
st.title('')
st.subheader(":blue[📁 Dataset used in model]")

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
train = pd.get_dummies(train, columns=cat_col)
st.write(train.head(3))
################################
#z코드
if st.button("⌨️ See code for KNN Regressor with Optuna"):
    st.code("""
    def objective(trial):
        # scaler list
        scalers = trial.suggest_categorical("scalers", ['minmax', 'standard', 'robust'])

        # scaler
        if scalers == "minmax":
            scaler = MinMaxScaler()
        elif scalers == "standard":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()

        dim_red = trial.suggest_categorical("dim_red", ["PCA", None])

        # pca 
        if dim_red == "PCA":
            pca_n_components=trial.suggest_int("pca_n_components", 2, x_train.shape[1]) 
            dimen_red_algorithm=PCA(n_components=pca_n_components)
        # (c) No dimensionality reduction option
        else:
            dimen_red_algorithm='passthrough'

        # 모델 하이퍼파라미터
        knn_n_neighbors=trial.suggest_int("knn_n_neighbors", 1, 19, 2)
        knn_metric=trial.suggest_categorical("knn_metric", ['euclidean', 'manhattan', 'minkowski'])
        knn_weights=trial.suggest_categorical("knn_weights", ['uniform', 'distance'])

        estimator=KNeighborsRegressor(n_neighbors=knn_n_neighbors, metric=knn_metric, weights=knn_weights)

        # pipeline
        pipeline = make_pipeline(scaler, dimen_red_algorithm, estimator)

        # cross-validation
        score = cross_val_score(pipeline, x_train, y_train, scoring='neg_mean_squared_log_error', cv= KFold(n_splits=5, shuffle=True, random_state=42))
        return -score.mean()

    # Optuna
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize",
                                sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    """)
####################################
#하이퍼파라미터 df
st.subheader(":blue[👍 Best Hyperparameters]")
hp_name = ["scalers", "dim_red", "n_neighbors", "metric", "weight"]
types = ["Scaler", "Pca", "int", "metric", "weight"]
ranges = ["""['minmax', 'standard', 'robust']""",
          """[2, x_train.shape[1]]""", 
          """np.arange(1, 19, 2)""",
          """['euclidean', 'manhattan', 'minkowski']""",
          """['uniform', 'distance']"""]
best_values = ['standard', "None", 3, "manhatan", "uniform"]
params = pd.DataFrame({'hp_name':hp_name,
                        "type": types,
                        'range':ranges,
                        'best_value':best_values})
st.write(params)

#######################################
st.title('')
st.subheader(":blue[Optuna Result]")
st.markdown('- 아래는 optimization 과정을 시각화한 것이다.\n'
            '- 사용한 scaler가 가장 영향력이 컸으며, neighbors 수가 metric이나 weight보다 주요했음을 알 수 있다.')

op = "bike_prediction/KNN/KNN Hour/op.png"
hp = "bike_prediction/KNN/KNN Hour/hi.png"
result = "bike_prediction/KNN/KNN Hour/pred.png"

# img2 = load_image(op)           
st.image(op)

# img3 = load_image(hp)           
st.image(hp)

st.title('')
st.subheader(":blue[Test data prediction result]")
st.markdown("- 실제값에 대한 예측값을 나타낸 산점도는 다음과 같다.")
st.image(result)

####################################################

