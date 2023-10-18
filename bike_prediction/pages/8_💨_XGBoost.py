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
st.title("💨 XGBoost 💨")
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

cv_score = 0.375
valid = 0.416
test = 0.453

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
cat_col = ["season", "Year","weather", "Day of week","Month","Day_info"] #Hour
for col in cat_col:
    train[col] = train[col].astype("category")
#target, drop, y
target_col = "count"
drop_cols = ["Unnamed: 0", "datetime", "workingday", "holiday", "Day", "Year", "sin_hour", "cos_hour",target_col] #"sin_hour", "cos_hour"
train = train.drop(drop_cols, axis=1)

cat_col = ["season", "weather", "Day of week", "Month", "Day_info","Hour"] #
train = pd.get_dummies(train, columns=cat_col)

st.subheader(":blue[📁 Dataset used in model]")
st.write(train.head(5))

#######################################
#model code
if st.button("⌨️ See code for XGB Regressor with Optuna"):
    st.code("""
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        }
        
        n_splits=5
        random_state=42
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_index, valid_index in kf.split(X=x_train, y=y_train):
            X_train, Y_train = x_train.iloc[train_index], y_train[train_index]
            X_valid, Y_valid = x_train.iloc[valid_index], y_train[valid_index]

            model = XGBRegressor(**param)
            model.fit(X_train, np.log(Y_train))

            xg_pred = model.predict(X_valid)
            scores.append(rmsle(np.log(Y_valid), xg_pred))
        
        return np.mean(scores)
    """)
####################################
st.subheader(":blue[👍 Best Hyperparameters]")

hp_name = ['max_depth',"learning_rate"
,'n_estimators','min_chid_weight','gamma','subsample'
,'colsample_bytree','reg_alpha','reg_lambda']
types = ['int',"float",'int','int',"float","float","float","float","float"]
ranges = ["""[1, 20]""",
    """[0.001, 1.0]""",
        """[50, 2000]""",
        """[1, 10]""",
        """[0.01, 1.0]""",
        """[0.01, 1.0]""",
        """[0.01, 1.0]""",
        """[0.01, 1.0]""",
        """[0.01, 1.0]"""
        ]
best_values = [20,0.18,1791,6,0.263,0.981,0.828,0.71,0.928]
params = pd.DataFrame({'hp_name':hp_name,
                        "type": types,
                        'range':ranges,
                        'best_value':best_values})
st.write(params)



######################################

op="bike_prediction/XGB/op.png"
hp = "bike_prediction/XGB/hp.png"
result = "bike_prediction/XGB/pred.png"

st.subheader(":blue[Optuna Result]")

# img2 = load_image(op)           
st.image(op)

# img3 = load_image(hp)           
st.image(hp)


#######
st.subheader(":blue[Test data prediction result]")       
st.image(result)

##############################
st.subheader(":blue[Shap Values]")
shap = "bike_prediction/XGB/xgb shap.png"
waterfall = "bike_prediction/XGB/xgb shap2.png"
### 맥이랑 window랑 백슬래쉬 반대키!!!
st.markdown("##### Overall Shap Values")
st.image(shap)

st.markdown("##### Model explanation for first row")       
st.image(waterfall)
