import streamlit as st
import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

st.set_page_config(
    layout="wide",
)
st.title("🗓️ TabNet 🗓️")
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

cv_score = 0.147
valid = 0.244
test = 0.556

v1, d1 = f1(cv_score, valid)
v2, d2 = f1(cv_score, test)


col1, col2, col3 = st.columns(3)
col1.metric("CV score", cv_score, "")
col2.metric("Test", valid, v1, delta_color=d1)
col3.metric("Kaggle (out of sample)", test, v2, delta_color=d2)
###############################
st.markdown('- 최근 딥러닝이 강세이지만, Tabular Data를 다루는 여러 대회를 보면 트리 기반 모델의 성능이 훨씬 좋은 것을 확인할 수 있다.\n'
            '- 하지만 2년 전 Kaggle에서 열린 신약개발 대회인 MOA에서 TabNet이 트리 기반 알고리즘을 제치고 상위권을 차지한 바 있다.\n'
            '- 하여 본 프로젝트에도 적용해보았으나 train set에 대한 overfitting 문제가 있었으며 트리 기반 모델에 비해 성능이 낮았다.\n'
            '- TabNet은 하이퍼파라미터가 많은 편이기 때문에, 이러한 결과는 튜닝과 최적화가 덜 이루어진 것에 기인했을 것으로 판단하였다.')

#dataset
@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train_eda.csv")
    return train

train= load_data()
train.drop("Unnamed: 0", axis=1, inplace=True)
columns = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'Day of week',
       'Month', 'Hour', 'Day_info']

st.subheader(":blue[📁 Dataset used in model]")
st.write(train[columns].head(5))
#######################################
st.markdown("- TabNet을 사용하기 위해서는 범주형 변수들의 label encoding 작업이 필요하다.")

# st.image("bike_prediction/TabNet/encoder.png")
######################################
if st.button("⌨️ See code for TabNet Label Encoding"):
    st.code("""class MultiColLabelEncoder:
    def __init__(self):
        self.encoder_dict = defaultdict(LabelEncoder)
    
    def fit_transform(self, X: pd.DataFrame, columns: list):
        if not isinstance(columns, list):
            columns = [columns]
        
        output = X.copy()
        output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x))

        return output
    
    def inverse_transform(self, X: pd.DataFrame, columns: list):
        if not isinstance(columns, list):
            columns = [columns]
        
        if not all(key in self.encoder_dict for key in columns):
            raise KeyError(f"at least one of {columns} is not encoded before")
        output = X.copy()
        try:
            output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))
        except ValueError:
            print(f"Need assignment for 'fit_transform' function")
            raise
        
        return output
    """)
class MultiColLabelEncoder:
    def __init__(self):
        self.encoder_dict = defaultdict(LabelEncoder)
    
    def fit_transform(self, X, columns):
        if not isinstance(columns, list):
            columns = [columns]
        
        output = X.copy()
        output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x))

        return output
    
    def inverse_transform(self, X, columns):
        if not isinstance(columns, list):
            columns = [columns]
        
        if not all(key in self.encoder_dict for key in columns):
            raise KeyError(f"at least one of {columns} is not encoded before")
        output = X.copy()
        try:
            output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))
        except ValueError:
            print(f"Need to use 'fit_transform' first")
            raise
        
        return output



# create an instance of the MultiColLabelEncoder class
mcle = MultiColLabelEncoder()
target_col = "count"
drop_cols = ["datetime", "workingday", "holiday", "Day", "Year", "Hour", target_col] # "sin_hour", "cos_hour"
train_x = train.drop(drop_cols, axis=1)
cat_cols = ["season", "weather", "Day of week", "Month", "Day_info"] # "Hour"
xtrain_t = mcle.fit_transform(train_x, columns=cat_cols)
st.write(xtrain_t.head())

#######################################

features = list(xtrain_t.columns)
cat_idxs = [i for i, f in enumerate(features) if f in cat_cols]
cat_dims = [len(xtrain_t[col].unique()) for col in cat_cols]

##########################################
st.markdown("- 모델의 input으로 범주형 변수의 이름, 행 index, 그리고 각 범주형 변수의 unique value 개수가 필요하다.")
st.code("""
features = list(xtrain_t.columns)
cat_idxs = [i for i, f in enumerate(features) if f in cat_cols]
cat_dims = [len(xtrain_t[col].unique()) for col in cat_cols]

print(features)
print(cat_idxs)
print(cat_dims)
""")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**전체 features**')
    features_d = pd.DataFrame(features)
    features_d.columns = ["Features"]
    st.write(features_d)
with col2:
    st.markdown('**범주형 변수의 index**')
    cat_idxs_d = pd.DataFrame(cat_idxs)
    cat_idxs_d.index = ["season", "weather", "Day of week", "Month", "Day_info"]
    cat_idxs_d.columns = ["index"]
    st.write(cat_idxs_d)
with col3:
    st.markdown('**범주형 변수의 unique value 개수**')
    cat_dims_d = pd.DataFrame(cat_dims)
    cat_dims_d.index = ["season", "weather", "Day of week", "Month", "Day_info"]
    cat_dims_d.columns = ["count"]
    st.write(cat_dims_d)

################################################33
#model code
st.subheader(":blue[Optuna Code]")
if st.button("⌨️ See code for TabNet with Optuna"):
    st.code("""
    def Objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        cat_emb_dim = trial.suggest_int("cat_emb_dim", 1, 10)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                         lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                         mask_type=mask_type, n_shared=n_shared,
                         scheduler_params=dict(mode="min",
                                               patience=trial.suggest_int("patienceScheduler",low=3,high=10),
                                               min_lr=1e-5,
                                               factor=0.5,),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=0,
                         cat_idxs = cat_idxs,
                         cat_dims = cat_dims,
                         cat_emb_dim=cat_emb_dim
                         )
        n_splits=5
        random_state=42
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        #local_train, local_valid를 5번 만들어서 수행
        for train_index, valid_index in kf.split(X=xtrain_t, y=y_train):
            X_train, Y_train = xtrain_t.iloc[train_index].values, np.array(y_train[train_index]).reshape(-1, 1)
            X_valid, Y_valid = xtrain_t.iloc[valid_index].values, np.array(y_train[valid_index]).reshape(-1, 1)

            regressor = TabNetRegressor(**tabnet_params)
            regressor.fit(X_train=X_train, y_train=Y_train,
                      eval_set=[(X_valid, Y_valid)],
                      patience=trial.suggest_int("patience",low=15,high=30), max_epochs=trial.suggest_int('epochs', 1, 100),
                      eval_metric=["rmsle"])

            scores.append(regressor.best_cost)

        avg = np.mean(scores)
        return avg

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="TabNet",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(Objective, n_trials=20)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    """)

#####################################
st.title('')
st.subheader(":blue[Optuna Result]")
st.image("bike_prediction/TabNet/newplot (4).png")
st.image("bike_prediction/TabNet/newplot (5).png")

######################################
